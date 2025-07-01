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
   ```

2. Set up the virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file by copying from `.env.example`:
   ```bash
   cp .env.example .env
   ```

5. Add your **Telegram Bot Token** and **Hugging Face Token** to the `.env` file.

6. Run the bot:
   ```bash
   python mcp_tg_bot.py
   ```

## Usage

- Start the bot in Telegram by messaging it directly or by using the `/start` command.
- You can ask questions related to the HODL ecosystem (e.g., "\$HODL tokenomics", "How do BNB rewards work?", "What are NFTs in HODL?").
- The bot uses **Retrieval-Augmented Generation (RAG)** to enhance answers by retrieving relevant context from ChromaDB and generating accurate responses using the MCP model.

## Development

To add new features or fix bugs:

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```

2. Make changes and test locally.

3. Push your changes:
   ```bash
   git push origin feature/your-feature
   ```

4. Open a pull request for review.

## License

This project is licensed under the MIT License.
