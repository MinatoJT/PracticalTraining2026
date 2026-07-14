param(
    [string]$Python = "C:\anaconda\python.exe",
    [string]$Revision = "v0.1.2",
    [string]$EvalModel = "deepseek-v4-flash",
    [string]$ExistingRunRoot = "",
    [switch]$SkipBaseline,
    [switch]$SkipAnchorOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$gitRoot = Split-Path -Parent $projectRoot
$sampleIds = Join-Path $projectRoot "UI\outputs\regression_diagnostics\20260713_182525_+0800\sample_ids.txt"
$baselineHead = "6d7bb37"
$stamp = (Get-Date -Format "yyyyMMdd_HHmmss_zzz").Replace(":", "")
$runRoot = if ([string]::IsNullOrWhiteSpace($ExistingRunRoot)) {
    Join-Path $projectRoot "UI\outputs\repair_runs\$stamp"
} else {
    [IO.Path]::GetFullPath($ExistingRunRoot)
}

function Read-SecretEnvironment([string]$Name, [string]$Prompt) {
    if (-not [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($Name, "Process"))) { return }
    $secret = Read-Host $Prompt -AsSecureString
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
    try {
        [Environment]::SetEnvironmentVariable(
            $Name,
            [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer),
            "Process"
        )
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
}

function Invoke-RepairRun([string]$Name, [hashtable]$Switches) {
    foreach ($entry in $Switches.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, [string]$entry.Value, "Process")
    }
    $outputDir = Join-Path $runRoot $Name
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    Write-Host "[$Name] Running the fixed 10-sample evaluation..."
    & $Python -B (Join-Path $projectRoot "UI\run_eval.py") `
        --task task1 --agent task1kg --num-conversations 10 `
        --display-conversations 10 --eval-model $EvalModel `
        --revision $Revision --sample-ids-file $sampleIds `
        --output-dir $outputDir --no-progress
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

if (-not (Test-Path -LiteralPath $Python)) { throw "Python was not found: $Python" }
if (-not (Test-Path -LiteralPath $sampleIds)) { throw "Sample ID file was not found: $sampleIds" }
Read-SecretEnvironment "DEEPSEEK_API_KEY" "Enter DeepSeek API Key (hidden)"
Read-SecretEnvironment "QWEN_VL_API_KEY" "Enter Qwen Vision API Key (hidden)"

$env:CRAGMM_CACHE_DIR = Join-Path $projectRoot "Dataset"
$env:HF_HOME = Join-Path $projectRoot "Dataset\hf_home"
$env:HF_DATASETS_CACHE = Join-Path $projectRoot "Dataset\hf_datasets"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $projectRoot "Dataset\hf_hub"
$env:HF_XET_CACHE = Join-Path $projectRoot "Dataset\hf_xet"
$env:TRANSFORMERS_CACHE = Join-Path $projectRoot "Dataset\transformers"
$env:SENTENCE_TRANSFORMERS_HOME = Join-Path $projectRoot "Dataset\sentence_transformers"
$env:CRAG_CACHE_DIR = Join-Path $projectRoot "Dataset\crag_images"
$env:CRAG_WEBSEARCH_CACHE_DIR = Join-Path $projectRoot "Dataset\crag_web_search"
$env:QWEN_VL_MODEL = "qwen3.5-omni-plus"
$env:QWEN_VL_RERANK_MODEL = "qwen3.5-omni-plus"
$env:QWEN_VL_RERANK_MAX_TOKENS = "4096"
$env:QWEN_VL_TIMEOUT = "45"
$env:QWEN_VL_MAX_RETRIES = "0"
if ([string]::IsNullOrWhiteSpace($env:QWEN_VL_BASE_URL)) {
    $env:QWEN_VL_BASE_URL = "https://ws-xlxf4xrlgnfbboal.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
}
$env:VISION_RERANK_MODE = "pure_visual"
$env:VISION_CACHE_ENABLED = "0"
$env:ANSWER_RELIABILITY_ENABLED = "0"
$env:VISUAL_VERIFIER_ENABLED = "0"
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

# Stage A uses a temporary worktree at the accepted HEAD and does not touch current changes.
if (-not $SkipBaseline) {
$baselineWorktree = Join-Path $env:TEMP "cragmm_repair_head_$stamp"
$resolvedTemp = [IO.Path]::GetFullPath($env:TEMP).TrimEnd('\')
$resolvedWorktree = [IO.Path]::GetFullPath($baselineWorktree)
if (-not $resolvedWorktree.StartsWith($resolvedTemp, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Baseline worktree is outside TEMP: $resolvedWorktree"
}

try {
    & git -C $gitRoot worktree add --detach $baselineWorktree $baselineHead
    if ($LASTEXITCODE -ne 0) { throw "Could not create the HEAD baseline worktree" }
    $baselineProject = Join-Path $baselineWorktree (Split-Path -Leaf $projectRoot)
    $baselineOut = Join-Path $runRoot "A_head_baseline"
    New-Item -ItemType Directory -Force -Path $baselineOut | Out-Null
    $env:VISION_ENABLED = "1"
    $env:VISION_RERANK_ENABLED = "1"
    $env:EVIDENCE_RETRY_ENABLED = "0"
    $env:TASK1_DEBUG_PATH = Join-Path $baselineOut "trace.jsonl"

    Write-Host "[A_head_baseline] Running first 10 rows from $baselineHead; their ID set matches sample_ids.txt."
    Push-Location $baselineProject
    try {
        & $Python -B (Join-Path $baselineProject "UI\run_eval.py") `
            --task task1 --agent task1kg --num-conversations 10 `
            --display-conversations 10 --eval-model $EvalModel `
            --revision $Revision --no-progress
        if ($LASTEXITCODE -ne 0) { throw "A_head_baseline failed with exit code $LASTEXITCODE" }
        $legacyOut = Join-Path $baselineProject "UI\outputs\task1"
        foreach ($name in @("scores_dictionary.json", "turn_evaluation_results_all.csv", "conversation_evaluation_results.csv")) {
            $source = Join-Path $legacyOut $name
            if (Test-Path -LiteralPath $source) {
                Copy-Item -LiteralPath $source -Destination (Join-Path $baselineOut $name) -Force
            }
        }
        $allTurns = Join-Path $baselineOut "turn_evaluation_results_all.csv"
        if (Test-Path -LiteralPath $allTurns) {
            Copy-Item -LiteralPath $allTurns -Destination (Join-Path $baselineOut "fixed_sample_results.csv") -Force
        }
    }
    finally { Pop-Location }

    @{
        stage = "A_head_baseline"; head = (& git -C $baselineWorktree rev-parse HEAD).Trim()
        python = $Python; revision = $Revision; eval_model = $EvalModel
        qwen_anchor_model = $env:QWEN_VL_MODEL; qwen_rerank_model = $env:QWEN_VL_RERANK_MODEL
        vision_enabled = $env:VISION_ENABLED; vision_rerank_enabled = $env:VISION_RERANK_ENABLED
        answer_reliability_enabled = $env:ANSWER_RELIABILITY_ENABLED
        visual_verifier_enabled = $env:VISUAL_VERIFIER_ENABLED
        evidence_retry_enabled = $env:EVIDENCE_RETRY_ENABLED; cache_enabled = $env:VISION_CACHE_ENABLED
        deepseek_key_present = -not [string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY)
        qwen_key_present = -not [string]::IsNullOrWhiteSpace($env:QWEN_VL_API_KEY)
        selection_note = "dataset first 10; ID set verified equal to sample_ids.txt"
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $baselineOut "config_snapshot.json") -Encoding UTF8
    "HEAD baseline: $Python -B UI/run_eval.py --task task1 --agent task1kg --num-conversations 10 --display-conversations 10 --eval-model $EvalModel --revision $Revision --no-progress" |
        Set-Content -LiteralPath (Join-Path $baselineOut "run_command.txt") -Encoding UTF8
}
finally {
    if (Test-Path -LiteralPath $baselineWorktree) { & git -C $gitRoot worktree remove --force $baselineWorktree }
}
}

if (-not $SkipAnchorOnly) {
    Invoke-RepairRun "B_anchor_only" @{
        VISION_ENABLED = "1"; VISION_RERANK_ENABLED = "0"; EVIDENCE_RETRY_ENABLED = "0"
    }
}
Invoke-RepairRun "C_pure_visual_rerank" @{
    VISION_ENABLED = "1"; VISION_RERANK_ENABLED = "1"; EVIDENCE_RETRY_ENABLED = "0"
}
Invoke-RepairRun "D_evidence_retry" @{
    VISION_ENABLED = "1"; VISION_RERANK_ENABLED = "1"; EVIDENCE_RETRY_ENABLED = "1"
}

$runRoot | Set-Content -LiteralPath (Join-Path $projectRoot "UI\outputs\latest_repair_run.txt") -Encoding UTF8
Write-Host "A/B/C/D completed. Output directory: $runRoot"
