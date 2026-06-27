param(
    [int]$SeqLen = 24,
    [int]$PredLen = 12,
    [int]$LimitPerDataset = 0,
    [int]$StartIndex = 0,
    [int]$EndIndex = 0,
    [int]$MaxTokens = 320,
    [int]$SaveEvery = 25,
    [double]$Sleep = 0.2,
    [switch]$Resume,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectDir

if (-not $env:ECNU_LLM_API_KEY) {
    throw "ECNU_LLM_API_KEY is not set. Set it in the current shell before running this script."
}
if (-not $env:ECNU_LLM_BASE_URL) {
    $env:ECNU_LLM_BASE_URL = "https://chat.ecnu.edu.cn/open/api/v1"
}
if (-not $env:ECNU_LLM_MODEL) {
    $env:ECNU_LLM_MODEL = "ecnu-plus"
}

$datasets = @(
    @{
        Name = "Algriculture"
        Input = "data/Algriculture/US_RetailBroilerComposite_Month.csv"
        Context = "US monthly retail broiler composite price time series."
        Meaning = "OT=US retail broiler composite price; higher values indicate higher poultry market prices."
    },
    @{
        Name = "Climate"
        Input = "data/Climate/US_precipitation_month.csv"
        Context = "US monthly precipitation time series."
        Meaning = "OT=US monthly precipitation amount; higher values indicate wetter conditions."
    },
    @{
        Name = "Economy"
        Input = "data/Economy/US_TradeBalance_Month.csv"
        Context = "US monthly trade-balance time series."
        Meaning = "OT=US trade balance in the dataset units; larger values indicate a higher trade-balance level."
    },
    @{
        Name = "Energy"
        Input = "data/Energy/US_GasolinePrice_Week.csv"
        Context = "US weekly gasoline-price time series."
        Meaning = "OT=US retail gasoline price in dollars per gallon; higher values indicate more expensive gasoline."
    },
    @{
        Name = "Public_Health"
        Input = "data/Public_Health/US_FLURATIO_Week.csv"
        Context = "US weekly influenza-ratio public-health time series."
        Meaning = "OT=US influenza ratio; higher values indicate stronger influenza activity."
    },
    @{
        Name = "Security"
        Input = "data/Security/US_FEMAGrant_Month.csv"
        Context = "US monthly FEMA grant time series."
        Meaning = "OT=US FEMA grant measure in the dataset units; higher values indicate larger grant activity."
    },
    @{
        Name = "SocialGood"
        Input = "data/SocialGood/Unadj_UnemploymentRate_ALL_processed.csv"
        Context = "US monthly unemployment-rate social-good time series."
        Meaning = "OT=US unemployment rate in percent; higher values indicate weaker labor-market conditions."
    },
    @{
        Name = "Traffic"
        Input = "data/Traffic/US_VMT_Month.csv"
        Context = "US monthly vehicle-miles-traveled traffic time series."
        Meaning = "OT=US vehicle miles traveled in the dataset units; higher values indicate more road traffic activity."
    }
)

New-Item -ItemType Directory -Force -Path "data/llm-generated" | Out-Null
New-Item -ItemType Directory -Force -Path "data/llm-generated/audit" | Out-Null

foreach ($dataset in $datasets) {
    $name = $dataset.Name
    $output = "data/llm-generated/${name}_H${SeqLen}_F${PredLen}_ecnu_llm.csv"
    $audit = "data/llm-generated/audit/${name}_H${SeqLen}_F${PredLen}_ecnu_llm_audit.jsonl"

    if ((Test-Path $output) -and (-not $Overwrite) -and (-not $Resume)) {
        Write-Host "Skipping existing output: $output"
        continue
    }

    $cmd = @(
        "scripts/build_ecnu_llm_text_dataset.py",
        "--input", $dataset.Input,
        "--output", $output,
        "--audit-output", $audit,
        "--domain", $name,
        "--domain-context", $dataset.Context,
        "--variable-meanings", $dataset.Meaning,
        "--seq-len", "$SeqLen",
        "--pred-len", "$PredLen",
        "--target-col", "OT",
        "--value-cols", "OT",
        "--text-column", "ECNU_LLM_Text",
        "--fact-column", "ECNU_LLM_Fact",
        "--pred-column", "ECNU_LLM_Pred",
        "--max-tokens", "$MaxTokens",
        "--sleep", "$Sleep",
        "--save-every", "$SaveEvery"
    )

    if ($LimitPerDataset -gt 0) {
        $cmd += @("--limit", "$LimitPerDataset")
    }
    if ($StartIndex -gt 0) {
        $cmd += @("--start-index", "$StartIndex")
    }
    if ($EndIndex -gt 0) {
        $cmd += @("--end-index", "$EndIndex")
    }
    if ($Resume) {
        $cmd += "--resume"
    }
    if ($Overwrite) {
        $cmd += "--overwrite"
    }

    Write-Host "Generating ECNU LLM text for $name -> $output"
    & python @cmd
}
