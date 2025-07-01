# HODLToken Telegram Bot

This project implements a Telegram bot for the **HODLToken** ecosystem, powered by the **MCP Model** (Mistral 7B) and **ChromaDB** for efficient question answering. The bot uses **Telegram API**, **MCP model** for text generation, and **ChromaDB** for storing and searching context data. This is a **Retrieval-Augmented Generation (RAG)** setup where the model retrieves relevant documents from ChromaDB and augments the response with relevant information.

## Features

- **Telegram Integration**: Responds to user queries about the HODLToken ecosystem.
- **MCP Model**: Uses the Mistral 7B model for natural language processing and generation.
- **ChromaDB**: Stores and retrieves contextual data, ensuring accurate and quick responses.
- **RAG-based Architecture**: Combines information retrieval from ChromaDB with text generation from the MCP model to answer user queries.
- **Data Population**: Load data from pre-processed chunks and populate ChromaDB for search.

## Setup

### Prerequisites

- Python 3.8+
- A Telegram Bot Token (create one via [BotFather](https://core.telegram.org/bots#botfather))
- Hugging Face Token (for accessing private models)
- Install [Playwright](https://playwright.dev/) (if using web crawling)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hodltoken-tg-bot.git
   cd hodltoken-tg-bot
