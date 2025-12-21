# Git Usage Guide

## What Has Been Set Up

✅ **Git repository initialized** - Your code is now version controlled  
✅ **Comprehensive .gitignore** - Excludes checkpoints, datasets, logs, and other unnecessary files  
✅ **Initial commit created** - All your current code is saved as the first version

## Basic Git Commands

### View Changes
```bash
git status                    # See what files have changed
git diff                      # See detailed changes in files
git log --oneline            # See commit history
```

### Save Your Work (Commit)
```bash
# Step 1: Stage files you want to save
git add <filename>           # Add specific file
git add .                    # Add all changed files

# Step 2: Commit with a message
git commit -m "Description of what you changed"
```

### Recover Previous Versions

#### View Previous Versions
```bash
git log                      # See all commits with details
git log --oneline           # Compact view
git show <commit-hash>      # See what changed in a specific commit
```

#### Recover a File from Previous Version
```bash
# Find the commit hash from git log, then:
git checkout <commit-hash> -- <filename>
```

#### Recover Entire Project to Previous Version
```bash
# View commits first
git log --oneline

# Go back to a specific commit (creates detached HEAD - be careful!)
git checkout <commit-hash>

# To go back to latest version:
git checkout master
```

#### Create a Branch from Previous Version (Safer)
```bash
# Create a new branch from an old commit
git checkout -b recovery-branch <commit-hash>

# Work on it, then merge back if needed
git checkout master
git merge recovery-branch
```

### Undo Changes (Before Committing)
```bash
git restore <filename>       # Discard changes to a file
git restore .                # Discard all uncommitted changes
```

### Undo Last Commit (Keep Changes)
```bash
git reset --soft HEAD~1     # Undo commit, keep changes staged
git reset HEAD~1            # Undo commit, keep changes unstaged
```

### See What Changed Between Versions
```bash
git diff <commit1> <commit2>           # Compare two commits
git diff HEAD~1 HEAD                   # Compare last commit with current
git diff <commit-hash> -- <filename>   # Compare specific file
```

## Recommended Workflow

1. **Make changes** to your code
2. **Check what changed**: `git status` and `git diff`
3. **Stage files**: `git add <files>` or `git add .`
4. **Commit**: `git commit -m "Clear description of changes"`
5. **Repeat** as you work

## Example: Recovering train_vit_cub.py from Yesterday

```bash
# 1. Find the commit from yesterday
git log --oneline --since="2 days ago" -- scripts/train_vit_cub.py

# 2. Recover that version
git checkout <commit-hash> -- scripts/train_vit_cub.py

# 3. Review the recovered file, then commit if you want to keep it
git commit -m "Restore train_vit_cub.py from previous version"
```

## Important Notes

- **Datasets and checkpoints are NOT tracked** (they're in .gitignore)
- **Only code and config files are versioned**
- **Always commit frequently** - it's like saving your work
- **Write clear commit messages** - they help you remember what changed

## Quick Reference

| Task | Command |
|------|---------|
| See what changed | `git status` |
| See detailed changes | `git diff` |
| Save changes | `git add . && git commit -m "message"` |
| View history | `git log --oneline` |
| Recover file | `git checkout <commit> -- <file>` |
| Undo uncommitted changes | `git restore <file>` |

