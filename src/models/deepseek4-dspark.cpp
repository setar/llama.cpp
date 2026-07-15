#include "models.h"

#include <stdexcept>

void llama_model_deepseek4_dspark::load_arch_hparams(llama_model_loader & ml) {
    llama_model_deepseek4::load_arch_hparams(ml);

    ml.get_key(LLM_KV_DSPARK_BLOCK_SIZE,     hparams.dspark_block_size);
    ml.get_key(LLM_KV_DSPARK_NOISE_TOKEN_ID, hparams.dspark_noise_token_id);
    ml.get_key(LLM_KV_DSPARK_MARKOV_RANK,    hparams.dspark_markov_rank);

    ml.get_arr_n(LLM_KV_DSPARK_TARGET_LAYER_IDS, hparams.dspark_n_target_layers);
    ml.get_arr  (LLM_KV_DSPARK_TARGET_LAYER_IDS, hparams.dspark_target_layers);

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

    const int64_t n_target     = hparams.dspark_n_target_layers;
    const int64_t markov_rank  = hparams.dspark_markov_rank;

    dspark_main_proj   = create_tensor(tn(LLM_TENSOR_DSPARK_MAIN_PROJ,   "weight"), {n_target * n_embd, n_embd}, 0);
    dspark_main_norm   = create_tensor(tn(LLM_TENSOR_DSPARK_MAIN_NORM,   "weight"), {n_embd}, 0);
    dspark_markov_embd = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_EMBD, "weight"), {markov_rank, n_vocab}, 0);
    dspark_markov_head = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_HEAD, "weight"), {markov_rank, n_vocab}, 0);
    dspark_conf_head   = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_HEAD,   "weight"), {n_embd + markov_rank}, 0);
}

std::unique_ptr<llm_graph_context> llama_model_deepseek4_dspark::build_arch_graph(const llm_graph_params & params) const {
    GGML_UNUSED(params);
    // draft graph lands with the speculative-decoding integration
    throw std::runtime_error("deepseek4-dspark graph is not implemented yet");
}
