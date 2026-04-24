# Skills

A fork of [huggingface/skills](https://huggingface.co/skills) — a collection of AI skills and agents for various tasks.

## Overview

This repository contains a curated set of skills that can be used with AI assistants and coding tools. Skills are modular, composable units that extend the capabilities of AI agents.

> **Personal note:** I'm using this fork primarily to experiment with custom coding and code-review skills for my own projects. The upstream repo is great but I wanted a sandbox to iterate faster.

## Structure

```
.
├── .claude-plugin/          # Claude AI plugin configuration
│   ├── plugin.json          # Plugin metadata and settings
│   └── marketplace.json     # Marketplace listing configuration
├── .cursor-plugin/          # Cursor IDE plugin configuration
│   ├── plugin.json          # Plugin metadata and settings
│   └── marketplace.json     # Marketplace listing configuration
├── .github/
│   └── workflows/
│       ├── generate-agents.yml          # CI: Auto-generate agent definitions
│       ├── push-evals-leaderboard.yml   # CI: Push evaluation results
│       └── push-hackers-leaderboard.yml # CI: Push hacker leaderboard results
└── README.md
```

## Getting Started

### Using with Claude

1. Install the Claude plugin by referencing `.claude-plugin/plugin.json`
2. Browse available skills in `.claude-plugin/marketplace.json`

### Using with Cursor

1. Install the Cursor plugin by referencing `.cursor-plugin/plugin.json`
2. Browse available skills in `.cursor-plugin/marketplace.json`

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork this repository
2. Create a feature branch (`git checkout -b feat/my-new-skill`)
3. Commit your changes (`git commit -m 'feat: add my new skill'`)
4. Push to the branch (`git push origin feat/my-new-skill`)
5. Open a Pull Request

### Adding a New Skill

- Skills should be self-contained and focused on a single task
- Include a clear description and usage examples
- Add appropriate tests and evaluation criteria
- Update the marketplace listings in both `.claude-plugin/` and `.cursor-plugin/`

## Leaderboards

This project maintains two leaderboards:

- **Evals Leaderboard**: Tracks performance of skills on standardized benchmarks
- **Hackers Leaderboard**: Tracks community contributions and improvements

Leaderboards are automatically updated via GitHub Actions workflows.

## Security

Please review our [Security Policy](.github/SECURITY.md) before reporting vulnerabilities.

## License

This project is licensed under the Apache 2.0 License — see the [LICENSE](LICENSE) file for details.
