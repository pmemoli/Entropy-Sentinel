presets=(
    "gsm8k" "test" "microsoft/Phi-3.5-mini-instruct" "phi3-gsm-test"
    "gsm8k" "train" "microsoft/Phi-3.5-mini-instruct" "phi3-gsm-train"
    "math" "test" "microsoft/Phi-3.5-mini-instruct" "phi3-mathhendrycks-test"
    "math" "train" "microsoft/Phi-3.5-mini-instruct" "phi3-mathhendrycks-train"
    "svamp" "test" "microsoft/Phi-3.5-mini-instruct" "phi3-svamp-test"
    "gsm8ksymbolic" "test" "microsoft/Phi-3.5-mini-instruct" "phi3-gsmsymbolic-test"
    "livemathbench" "test" "microsoft/Phi-3.5-mini-instruct" "phi3-livemathbench-test"
    "gpqa" "test" "microsoft/Phi-3.5-mini-instruct" "phi3-gpqa-test"
    "scibench" "test" "microsoft/Phi-3.5-mini-instruct" "phi3-scibench-test"
    "theoremqa" "test" "microsoft/Phi-3.5-mini-instruct" "phi3-theoremqa-test"
    "olympiadbench" "test" "microsoft/Phi-3.5-mini-instruct" "phi3-olympiadbench-test"
    "matscibench" "test" "microsoft/Phi-3.5-mini-instruct" "matscibench"

    "gsm8k" "test" "Qwen/Qwen3-4B-Instruct-2507" "qwen3-gsm-test"
    "gsm8k" "train" "Qwen/Qwen3-4B-Instruct-2507" "qwen3-gsm-train"
    "mathhendrycks" "test" "Qwen/Qwen3-4B-Instruct-2507" "qwen3-mathhendrycks-test"
    "mathhendrycks" "train" "Qwen/Qwen3-4B-Instruct-2507" "qwen3-mathhendrycks-train"
    "svamp" "test" "Qwen/Qwen3-4B-Instruct-2507" "qwen3-svamp-test"
    "gsm8ksymbolic" "test" "Qwen/Qwen3-4B-Instruct-2507" "qwen3-gsmsymbolic-test"
    "livemathbench" "test" "Qwen/Qwen3-4B-Instruct-2507" "qwen3-livemathbench-test"
    "gpqa" "test" "Qwen/Qwen3-4B-Instruct-2507" "qwen3-gpqa-test"
    "scibench" "test" "Qwen/Qwen3-4B-Instruct-2507" "qwen3-scibench-test"
    "theoremqa" "test" "Qwen/Qwen3-4B-Instruct-2507" "qwen3-theoremqa-test"
    "olympiadbench" "test" "Qwen/Qwen3-4B-Instruct-2507" "qwen3-olympiadbench-test"
    "matscibench" "test" "Qwen/Qwen3-4B-Instruct-2507" "qwen3-matscibench-test"
    
    "gsm8k" "test" "Qwen/Qwen3-8B" "qwen3-8b-gsm-test"
    "gsm8k" "train" "Qwen/Qwen3-8B" "qwen3-8b-gsm-train"
    "mathhendrycks" "test" "Qwen/Qwen3-8B" "qwen3-8b-mathhendrycks-test"
    "mathhendrycks" "train" "Qwen/Qwen3-8B" "qwen3-8b-mathhendrycks-train"
    "svamp" "test" "Qwen/Qwen3-8B" "qwen3-8b-svamp-test"
    "gsm8ksymbolic" "test" "Qwen/Qwen3-8B" "qwen3-8b-gsmsymbolic-test"
    "livemathbench" "test" "Qwen/Qwen3-8B" "qwen3-8b-livemathbench-test"
    "gpqa" "test" "Qwen/Qwen3-8B" "qwen3-8b-gpqa-test"
    "scibench" "test" "Qwen/Qwen3-8B" "qwen3-8b-scibench-test"
    "theoremqa" "test" "Qwen/Qwen3-8B" "qwen3-8b-theoremqa-test"
    "olympiadbench" "test" "Qwen/Qwen3-8B" "qwen3-8b-olympiadbench-test"
    "matscibench" "test" "Qwen/Qwen3-8B" "qwen3-8b-matscibench-test"

    "gsm8k" "test" "mistralai/Ministral-3-3B-Instruct-2512" "ministral3-3b-gsm-test"
    "gsm8k" "train" "mistralai/Ministral-3-3B-Instruct-2512" "ministral3-3b-gsm-train"
    "mathhendrycks" "test" "mistralai/Ministral-3-3B-Instruct-2512" "ministral3-3b-mathhendrycks-test"
    "mathhendrycks" "train" "mistralai/Ministral-3-3B-Instruct-2512" "ministral3-3b-mathhendrycks-train"
    "svamp" "test" "mistralai/Ministral-3-3B-Instruct-2512" "ministral3-3b-svamp-test"
    "gsm8ksymbolic" "test" "mistralai/Ministral-3-3B-Instruct-2512" "ministral3-3b-gsmsymbolic-test"
    "livemathbench" "test" "mistralai/Ministral-3-3B-Instruct-2512" "ministral3-3b-livemathbench-test"
    "gpqa" "test" "mistralai/Ministral-3-3B-Instruct-2512" "ministral3-3b-gpqa-test"
    "scibench" "test" "mistralai/Ministral-3-3B-Instruct-2512" "ministral3-3b-scibench-test"
    "olympiadbench" "test" "mistralai/Ministral-3-3B-Instruct-2512" "ministral3-3b-olympiadbench-test"
    "theoremqa" "test" "mistralai/Ministral-3-3B-Instruct-2512" "ministral3-3b-theoremqa-test"
    "matscibench" "test" "mistralai/Ministral-3-3B-Instruct-2512" "ministral3-3b-matscibench-test"
    
    "gsm8k" "test" "mistralai/Ministral-3-8B-Instruct-2512" "ministral3-gsm-test"
    "gsm8k" "train" "mistralai/Ministral-3-8B-Instruct-2512" "ministral3-gsm-train"
    "mathhendrycks" "test" "mistralai/Ministral-3-8B-Instruct-2512" "ministral3-mathhendrycks-test"
    "mathhendrycks" "train" "mistralai/Ministral-3-8B-Instruct-2512" "ministral3-mathhendrycks-train"
    "svamp" "test" "mistralai/Ministral-3-8B-Instruct-2512" "ministral3-svamp-test"
    "gsm8ksymbolic" "test" "mistralai/Ministral-3-8B-Instruct-2512" "ministral3-gsmsymbolic-test"
    "livemathbench" "test" "mistralai/Ministral-3-8B-Instruct-2512" "ministral3-livemathbench-test"
    "gpqa" "test" "mistralai/Ministral-3-8B-Instruct-2512" "ministral3-gpqa-test"
    "scibench" "test" "mistralai/Ministral-3-8B-Instruct-2512" "ministral3-scibench-test"
    "olympiadbench" "test" "mistralai/Ministral-3-8B-Instruct-2512" "ministral3-olympiadbench-test"
    "theoremqa" "test" "mistralai/Ministral-3-8B-Instruct-2512" "ministral3-theoremqa-test"
    "matscibench" "test" "mistralai/Ministral-3-8B-Instruct-2512" "ministral3-matscibench-test"
    
    "gsm8k" "test" "meta-llama/Llama-3.1-8B-Instruct" "llama3-gsm-test"
    "gsm8k" "train" "meta-llama/Llama-3.1-8B-Instruct" "llama3-gsm-train"
    "mathhendrycks" "test" "meta-llama/Llama-3.1-8B-Instruct" "llama3-mathhendrycks-test"
    "mathhendrycks" "train" "meta-llama/Llama-3.1-8B-Instruct" "llama3-mathhendrycks-train"
    "svamp" "test" "meta-llama/Llama-3.1-8B-Instruct" "llama3-svamp-test"
    "gsm8ksymbolic" "test" "meta-llama/Llama-3.1-8B-Instruct" "llama3-gsmsymbolic-test"
    "livemathbench" "test" "meta-llama/Llama-3.1-8B-Instruct" "llama3-livemathbench-test"
    "gpqa" "test" "meta-llama/Llama-3.1-8B-Instruct" "llama3-gpqa-test"
    "scibench" "test" "meta-llama/Llama-3.1-8B-Instruct" "llama3-scibench-test"
    "theoremqa" "test" "meta-llama/Llama-3.1-8B-Instruct" "llama3-theoremqa-test"
    "olympiadbench" "test" "meta-llama/Llama-3.1-8B-Instruct" "llama3-olympiadbench-test"
    "matscibench" "test" "meta-llama/Llama-3.1-8B-Instruct" "llama3-matscibench-test"

    "gsm8k" "test" "google/gemma-3-4b-it" "gemma3-4b-gsm-test"
    "gsm8k" "train" "google/gemma-3-4b-it" "gemma3-4b-gsm-train"
    "mathhendrycks" "test" "google/gemma-3-4b-it" "gemma3-4b-mathmathhendrycks-test"
    "mathhendrycks" "train" "google/gemma-3-4b-it" "gemma3-4b-mathhendrycks-train"
    "svamp" "test" "google/gemma-3-4b-it" "gemma3-4b-svamp-test"
    "gsm8ksymbolic" "test" "google/gemma-3-4b-it" "gemma3-4b-gsmsymbolic-test"
    "livemathbench" "test" "google/gemma-3-4b-it" "gemma3-4b-livemathbench-test"
    "gpqa" "test" "google/gemma-3-4b-it" "gemma3-4b-gpqa-test"
    "scibench" "test" "google/gemma-3-4b-it" "gemma3-4b-scibench-test"
    "theoremqa" "test" "google/gemma-3-4b-it" "gemma3-4b-theoremqa-test"
    "olympiadbench" "test" "google/gemma-3-4b-it" "gemma3-4b-olympiadbench-test"
    "matscibench" "test" "google/gemma-3-4b-it" "gemma3-4b-matscibench-test"
    
    "gsm8k" "test" "google/gemma-3-12b-it" "gemma3-gsm-test"
    "gsm8k" "train" "google/gemma-3-12b-it" "gemma3-gsm-train"
    "mathhendrycks" "test" "google/gemma-3-12b-it" "gemma3-mathmathhendrycks-test"
    "mathhendrycks" "train" "google/gemma-3-12b-it" "gemma3-mathhendrycks-train"
    "svamp" "test" "google/gemma-3-12b-it" "gemma3-svamp-test"
    "gsm8ksymbolic" "test" "google/gemma-3-12b-it" "gemma3-gsmsymbolic-test"
    "livemathbench" "test" "google/gemma-3-12b-it" "gemma3-livemathbench-test"
    "gpqa" "test" "google/gemma-3-12b-it" "gemma3-gpqa-test"
    "scibench" "test" "google/gemma-3-12b-it" "gemma3-scibench-test"
    "theoremqa" "test" "google/gemma-3-12b-it" "gemma3-theoremqa-test"
    "olympiadbench" "test" "google/gemma-3-12b-it" "gemma3-olympiadbench-test"
    "matscibench" "test" "google/gemma-3-12b-it" "gemma3-matscibench-test"
    
    "gsm8k" "test" "openai/gpt-oss-20b" "oss-gsm-test"
    "gsm8k" "train" "openai/gpt-oss-20b" "oss-gsm-train"
    "mathhendrycks" "test" "openai/gpt-oss-20b" "oss-mathhendrycks-test"
    "mathhendrycks" "train" "openai/gpt-oss-20b" "oss-mathhendrycks-train"
    "svamp" "test" "openai/gpt-oss-20b" "oss-svamp-test"
    "gsm8ksymbolic" "test" "openai/gpt-oss-20b" "oss-gsmsymbolic-test"
    "livemathbench" "test" "openai/gpt-oss-20b" "oss-livemathbench-test"
    "gpqa" "test" "openai/gpt-oss-20b" "oss-gpqa-test"
    "scibench" "test" "openai/gpt-oss-20b" "oss-scibench-test"
    "theoremqa" "test" "openai/gpt-oss-20b" "oss-theoremqa-test"
    "olympiadbench" "test" "openai/gpt-oss-20b" "oss-olympiadbench-test"
    "matscibench" "test" "openai/gpt-oss-20b" "oss-matscibench-test"
)

for ((i = 0; i < ${#presets[@]}; i+=4)); do
    dataset_name=${presets[i]}     
    split=${presets[i+1]}          
    model_name=${presets[i+2]}     
    suite=${presets[i+3]}          

    echo "Processing suite: ${suite}"

    python -m src.engine.store_activations \
      --dataset_name "${dataset_name}" \
      --split "${split}" \
      --model_name "${model_name}" \
      --suite "${suite}" \
      --result_path "./src/data/runs" \
      --max_length 2048

    echo "Completed storing activations for ${suite}."
done
