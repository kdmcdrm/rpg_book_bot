# RPG Rules Bot
This is a simple tool to look up game rules using a vector store and provide
them with a snarky interface.

## Installation
1. Install requirements `pip install -r requirements.txt`. Likely want to use an environment.
2. Copy env_example to .env and fill in the required elements.
3. Add the rulebook files to `./books` and update `./books/book_map.json` with name to filename mapping.
4. Run `process_rulebooks.py` to create the vector stores.
5. Run `rpg_book_bot.py` to bring up the interface.