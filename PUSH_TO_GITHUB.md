# Pushing this repository to GitHub (with Git LFS)

Step-by-step for **Windows**. Do these once per machine; the last three commands are the actual push.

## 0. Install prerequisites (once)

- **Git**: <https://git-scm.com/download/win> (Next → Next → Finish). After install, open a **new** PowerShell so `git` is on PATH.
- **Git LFS**: <https://git-lfs.com/> (or `git lfs install` after installing the LFS binary).

Verify:

```powershell
git --version
git lfs version
```

## 1. Register LFS filters (once per machine)

```powershell
git lfs install
```

## 2. Create an empty GitHub repository

On github.com → **New repository** → name e.g. `HB2204-condition-recognition` → choose **Public** (or Private) → **do NOT** tick "Add a README / .gitignore / license" (they already exist here) → Create.

Copy the repo URL shown (e.g. `https://github.com/<you>/HB2204-condition-recognition.git`).

## 3. In this folder, initialise, stage and commit

Open PowerShell in the repository folder:

```powershell
cd D:\Desktop\中国一拖\小论文1\HB2204_condition_recognition_github

git init
git add .gitattributes          # IMPORTANT: stage .gitattributes FIRST so LFS applies to data/
git add .
git commit -m "Initial release: HB2204 series-hybrid tractor condition recognition (physics-constrained PC-GMM, CAN-only vs Full features, INT8 edge deployment)"
git branch -M main
```

> If a filename contains spaces or special characters that git warns about, add them with `git add -A` and commit; the UTF-8 filenames are fine on modern git.

## 4. Push

```powershell
git remote add origin https://github.com/<you>/HB2204-condition-recognition.git
git push -u origin main
```

When prompted, authenticate with your GitHub credentials (or a Personal Access Token).

## 5. Verify

- On the GitHub repo page, `data/` files show as **"Stored with Git LFS"** (pointer text, not the 40 MB binary).
- `git lfs ls-files` locally lists the tracked data files.

## Notes

- **Quota**: GitHub gives 1 GB of LFS storage free; this repo needs ≈ 245 MB → comfortably within limit. If you later exceed it, upgrade or move data to an external host (Zenodo/OSF) and add a download script.
- **Why LFS**: without it, cloning pulls 245 MB every time and the repo's object store grows with every data edit. LFS stores pointers in git and the real bytes on the host.
- If you already pushed without LFS and want to convert: `git lfs migrate import --include="data/*.csv,data/*.mat" --everything` then re-push (`git push --force` if needed on a fresh repo).
