# Reproducible developer environment
> A version-controlled setup that rebuilds a full development machine — packages, runtimes, editor and shell — from a few files.

**Tools:** Homebrew (Brewfile), conda, zsh, VS Code, Git, shell scripting

*The real setup repo is private (it contains machine-specific paths); this is a sanitized overview.*

## Why
Setting up a laptop by hand is slow and impossible to repeat exactly. Keeping the setup in Git means
a new machine — or a broken one — is one script away from ready.

## How it's organized
| Piece | What it holds |
|---|---|
| `Brewfile` | Every CLI tool and app, installed with `brew bundle` |
| `Shell/` | zsh config, aliases and backups of previous configs |
| `Python/` | Shared conda environments exported as `environment.yml` files |
| `VSCode/` | Extension list so the editor can be restored with one loop |
| `Git/` | Conventions: one folder for all repositories, sources only |
| Secrets helper | Loads API keys at runtime from a secure store instead of files — **nothing secret is ever committed** |
| `PLAN.md`, `CHANGELOG.md` | What's set up, what's paused and why, and a dated history of changes |

## Example (illustrative)
```ruby
# Brewfile (excerpt)
brew "git"
brew "gh"
brew "python@3.12"
brew "node"
cask "visual-studio-code"
```
```bash
brew bundle --file=Brewfile                      # install everything
conda env create -f Python/web-shared.environment.yml
xargs -n1 code --install-extension < VSCode/extensions.txt
```
