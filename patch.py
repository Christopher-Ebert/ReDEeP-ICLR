from patch_llama_model import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaMLP,
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
    LlamaDecoderLayer,
    LlamaPreTrainedModel,
    LlamaModel,
    LlamaForSequenceClassification,
    LlamaForQuestionAnswering,
    LlamaForTokenClassification
)

import transformers.models.llama.modeling_llama as ml


def patch():
    patch_list = [
        LlamaRMSNorm,
        LlamaRotaryEmbedding,
        LlamaLinearScalingRotaryEmbedding,
        LlamaDynamicNTKScalingRotaryEmbedding,
        LlamaMLP,
        #LlamaAttention,
        #LlamaFlashAttention2,
        #LlamaSdpaAttention,
        LlamaDecoderLayer,
        LlamaPreTrainedModel,
        LlamaModel,
        LlamaForSequenceClassification,
        LlamaForQuestionAnswering,
        LlamaForTokenClassification
    ]
    for obj in patch_list:
        if hasattr(ml, obj.__name__):
            setattr(ml, obj.__name__, obj)


if __name__ == "__main__":
    patch()
