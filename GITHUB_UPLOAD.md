# Uploading this repository to GitHub

## Option 1 — GitHub website

1. Sign in to GitHub and choose **New repository**.
2. Suggested repository name: `phosphosite-prediction-svm`.
3. Suggested description:

   `Protein phosphorylation-site prediction using one-hot and ESM-2 embeddings with protein-grouped linear SVMs.`

4. Keep the repository **Public** if you want to use it as a CV/portfolio link.
5. Do **not** initialise it with another README, `.gitignore`, or licence because these files are already included here.
6. Create the repository.
7. Open the new repository and choose **uploading an existing file**.
8. Upload the *contents* of this folder, preserving the directory structure.
9. Commit message: `Initial reproducible phosphosite SVM pipeline`.

## Option 2 — Git command line

From inside this repository folder:

```bash
git init
git add .
git commit -m "Initial reproducible phosphosite SVM pipeline"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/phosphosite-prediction-svm.git
git push -u origin main
```

## Recommended GitHub settings

After upload:

- Add repository topics: `bioinformatics`, `machine-learning`, `protein-language-model`, `phosphorylation`, `svm`, `esm2`.
- Pin the repository to your GitHub profile if it is one of your strongest research projects.
- In the repository **About** section, use the same short description shown above.

## CV link

A concise CV project bullet could be:

`Built protein-grouped linear SVM models using sequence one-hot and ESM-2 embeddings to predict phosphorylation sites from 21-aa protein windows.`

If space permits, add the research context separately rather than claiming that every part of the cleaned GitHub workflow was used unchanged during the original placement.
