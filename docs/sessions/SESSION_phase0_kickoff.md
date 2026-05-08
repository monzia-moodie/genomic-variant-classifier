(.venv) PS C:\Projects\genomic-variant-classifier> # 1. Confirm repo root and current branch
(.venv) PS C:\Projects\genomic-variant-classifier> Set-Location C:\Projects\genomic-variant-classifier
(.venv) PS C:\Projects\genomic-variant-classifier> git status
On branch main
Your branch is up to date with 'origin/main'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        docs/sessions/SESSION_phase0_kickoff.md

nothing added to commit but untracked files present (use "git add" to track)
(.venv) PS C:\Projects\genomic-variant-classifier> git log --oneline -5
e65011f (HEAD -> main, origin/main, origin/HEAD) docs(incident): split duplicates + structural variant multimapping
089b2ed feat(splits): gene-set-complement meta_train reconstruction
93be9ed (tag: run9a-baseline) Merge pull request #1 from monzia-moodie-repo-projects/run9a-prep
c1c0192 (origin/run9a-prep, run9a-prep) chore: pin pandas 2.3.3, add session-docs helper, drawio workspace setting
5be5c8f chore: gitignore .venv*, *.bak-*, session scratch files
(.venv) PS C:\Projects\genomic-variant-classifier> 
(.venv) PS C:\Projects\genomic-variant-classifier> # 2. Confirm what already exists (NOT assumed)
(.venv) PS C:\Projects\genomic-variant-classifier> Test-Path .\agent_layer\
True
(.venv) PS C:\Projects\genomic-variant-classifier> Test-Path .\agent_layer\__init__.py
False
(.venv) PS C:\Projects\genomic-variant-classifier> Test-Path .\agent_layer\shared_infra\
False
(.venv) PS C:\Projects\genomic-variant-classifier> Test-Path .\src\drift\
False
(.venv) PS C:\Projects\genomic-variant-classifier> Test-Path .\scripts\preflight_review.py
False
(.venv) PS C:\Projects\genomic-variant-classifier> Test-Path .\scripts\validate_docs.py
False
(.venv) PS C:\Projects\genomic-variant-classifier> Test-Path .\scripts\check_run_id_trailer.py
False
(.venv) PS C:\Projects\genomic-variant-classifier> Test-Path .\.pre-commit-config.yaml
False
(.venv) PS C:\Projects\genomic-variant-classifier> Test-Path .\docs\validated\
False
(.venv) PS C:\Projects\genomic-variant-classifier> Test-Path .\docs\hypotheses\
False
(.venv) PS C:\Projects\genomic-variant-classifier> Test-Path .\docs\incidents\
True
(.venv) PS C:\Projects\genomic-variant-classifier> 
(.venv) PS C:\Projects\genomic-variant-classifier> # 3. Confirm Python and venv
(.venv) PS C:\Projects\genomic-variant-classifier> .\.venv\Scripts\python.exe --version    # expect 3.12.x
Python 3.14.3
(.venv) PS C:\Projects\genomic-variant-classifier> .\.venv\Scripts\pip.exe list | Select-String -Pattern "pydantic|fastapi|aiosqlite|portalocker|httpx|evidently|nannyml|alibi|river"

fastapi                    0.135.2
httpx                      0.28.1
pydantic                   2.12.5
pydantic_core              2.41.5

(.venv) PS C:\Projects\genomic-variant-classifier> 
(.venv) PS C:\Projects\genomic-variant-classifier> # 4. Confirm GCS access
(.venv) PS C:\Projects\genomic-variant-classifier> gcloud config get-value project        # expect genomic-variant-prod
genomic-variant-prod


Updates are available for some Google Cloud CLI components.  To install them,
please run:
  $ gcloud components update

(.venv) PS C:\Projects\genomic-variant-classifier> gsutil ls gs://genomic-variant-prod-outputs/ | Select-Object -First 5
ServiceException: 401 Anonymous caller does not have storage.objects.list access to the Google Cloud Storage bucket. Permission 'storage.objects.list' denied on resource (or it may not exist).
(.venv) PS C:\Projects\genomic-variant-classifier> # Confirm what already exists in the agent layer (don't assume from memory)
(.venv) PS C:\Projects\genomic-variant-classifier> Test-Path .\agent_layer\
True
(.venv) PS C:\Projects\genomic-variant-classifier> Test-Path .\agent_layer\__init__.py
False
(.venv) PS C:\Projects\genomic-variant-classifier> Test-Path .\agent_layer\shared_infra\
False
(.venv) PS C:\Projects\genomic-variant-classifier> Test-Path .\src\drift\
False
(.venv) PS C:\Projects\genomic-variant-classifier> Test-Path .\scripts\preflight_review.py
False
(.venv) PS C:\Projects\genomic-variant-classifier> Test-Path .\scripts\validate_docs.py
False
(.venv) PS C:\Projects\genomic-variant-classifier> Test-Path .\scripts\check_run_id_trailer.py
False
(.venv) PS C:\Projects\genomic-variant-classifier> Test-Path .\.pre-commit-config.yaml
False
(.venv) PS C:\Projects\genomic-variant-classifier> Test-Path .\docs\validated\
False
(.venv) PS C:\Projects\genomic-variant-classifier> Test-Path .\docs\hypotheses\
False
(.venv) PS C:\Projects\genomic-variant-classifier> Test-Path .\docs\incidents\
True
(.venv) PS C:\Projects\genomic-variant-classifier> 
(.venv) PS C:\Projects\genomic-variant-classifier> # Python and venv
(.venv) PS C:\Projects\genomic-variant-classifier> .\.venv\Scripts\python.exe --version
Python 3.14.3
(.venv) PS C:\Projects\genomic-variant-classifier> 
(.venv) PS C:\Projects\genomic-variant-classifier> # GCS
(.venv) PS C:\Projects\genomic-variant-classifier> gcloud config get-value project
genomic-variant-prod
(.venv) PS C:\Projects\genomic-variant-classifier> gcloud storage ls gs://genomic-variant-prod-outputs/ | Select-Object -First 5
ERROR: (gcloud.storage.ls) gs://genomic-variant-prod-outputs not found: 404.
(.venv) PS C:\Projects\genomic-variant-classifier> 