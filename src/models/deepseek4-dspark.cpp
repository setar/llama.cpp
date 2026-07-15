#include "models.h"

#include "llama-kv-cache-dsv4.h"

#include <stdexcept>

void llama_model_deepseek4_dspark::load_arch_hparams(llama_model_loader & ml) {
    llama_model_deepseek4::load_arch_hparams(ml);

    ml.get_key(LLM_KV_DSPARK_BLOCK_SIZE,     hparams.dspark_block_size);
    ml.get_key(LLM_KV_DSPARK_NOISE_TOKEN_ID, hparams.dspark_noise_token_id);
    ml.get_key(LLM_KV_DSPARK_MARKOV_RANK,    hparams.dspark_markov_rank);

    if (!ml.get_arr(LLM_KV_DSPARK_TARGET_LAYER_IDS, target_layer_ids, false) || target_layer_ids.empty()) {
        throw std::runtime_error("deepseek4-dspark requires dspark.target_layer_ids in GGUF metadata");
    }

    // encoder consumes the concatenated hc-collapsed outputs of the target layers
    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * hparams.n_embd;

    // DSparkAttention runs the raw (uncompressed) attention path only
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (hparams.dsv4_compress_ratios[il] != 0) {
            throw std::runtime_error("deepseek4-dspark expects compress_ratio 0 on all draft blocks");
        }
    }
}

void llama_model_deepseek4_dspark::load_arch_tensors(llama_model_loader & ml) {
    llama_model_deepseek4::load_arch_tensors(ml);

    LLAMA_LOAD_LOCALS;

    const int64_t n_target    = (int64_t) target_layer_ids.size();
    const int64_t markov_rank = hparams.dspark_markov_rank;

    dspark_main_proj   = create_tensor(tn(LLM_TENSOR_DSPARK_MAIN_PROJ,   "weight"), {n_target * n_embd, n_embd}, 0);
    dspark_main_norm   = create_tensor(tn(LLM_TENSOR_DSPARK_MAIN_NORM,   "weight"), {n_embd}, 0);
    dspark_markov_embd = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_EMBD, "weight"), {markov_rank, n_vocab}, 0);
    dspark_markov_head = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_HEAD, "weight"), {markov_rank, n_vocab}, 0);
    dspark_conf_head   = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_HEAD,   "weight"), {n_embd + markov_rank}, 0);
}

// Encoder: main_hidden [n_target*n_embd, n] -> main_x = main_norm(main_proj(main_hidden))
llama_model_deepseek4_dspark::graph_enc::graph_enc(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    auto inp = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp_enc());

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp_enc(), n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * cur = inp->embd;
    cb(cur, "inp_main_hidden", -1);

    res->add_input(std::move(inp));

    cur = build_lora_mm(model.dspark_main_proj, cur);
    cb(cur, "dspark_main_proj", -1);

    cur = build_norm(cur, model.dspark_main_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "dspark_main_x", -1);

    ggml_set_output(cur);
    res->t_h_nextn = cur;

    ggml_build_forward_expand(gf, cur);
}

// Cache-fill: embd batch carries main_x rows; per block write kv_norm(wkv(main_x))
// with RoPE at the row's absolute position into the raw KV cache. No block compute —
// the reference DSparkAttention prefill path only fills the cache (model.py:763).
llama_model_deepseek4_dspark::graph_inject::graph_inject(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;

    ggml_tensor * inp_pos = build_inp_pos();

    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * main_x = inp->embd;
    cb(main_x, "inp_main_x", -1);

    res->add_input(std::move(inp));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        // mirror the raw-path kv processing of deepseek4 build_attention (ratio == 0)
        ggml_tensor * kv = build_lora_mm(layer.wkv, main_x);
        kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
        kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, n_tokens);
        cb(kv, "kv_norm", il);

        ggml_tensor * kv_nope = ggml_view_3d(ctx0, kv, n_embd_head_nope, 1, n_tokens,
                ggml_row_size(kv->type, n_embd_head),
                ggml_row_size(kv->type, n_embd_head),
                0);
        ggml_tensor * kv_pe = ggml_view_3d(ctx0, kv, n_embd_head_rope, 1, n_tokens,
                ggml_row_size(kv->type, n_embd_head),
                ggml_row_size(kv->type, n_embd_head),
                ggml_row_size(kv->type, n_embd_head_nope));
        kv_pe = ggml_rope_ext(ctx0, kv_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
                freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        cb(kv_pe, "kv_pe", il);
        kv = ggml_concat(ctx0, kv_nope, kv_pe, 0);
        cb(kv, "kv", il);

        ggml_tensor * k_rot = inp_attn->self_k_rot;
        if (k_rot) {
            kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
        }

        ggml_build_forward_expand(gf, kv);
        ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));
    }

    res->t_embd = main_x;

    ggml_build_forward_expand(gf, main_x);
}

std::unique_ptr<llm_graph_context> llama_model_deepseek4_dspark::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph_enc>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            if (params.ubatch.embd) {
                return std::make_unique<graph_inject>(*this, params);
            }
            // draft block: the deepseek4 graph as-is — raw SWA attention over the
            // 128-position window implements the reference ring semantics
            return std::make_unique<llama_model_deepseek4::graph>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    }
}
