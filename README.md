# Rules Master 9000
Have you ever forgotten a RPG rule and the Googled it, but you wished that Google would be mean to you?
I built this package to solve that problem (for some reason).

It uses Retrieval Augmented Generation, first searching for the most relevant sections of the rulebook
and then providing those to the LLM to provide the final rules answer. Here's an example:

![](\docs\example1.png)


## Installation
1. Install requirements `pip install -r requirements.txt`. Likely want to use an environment.
2. Copy env_example to .env and fill in the required elements.
3. Add the rulebook files to `./books` and update `./books/book_map.json` with name to filename mapping.
4. Run `process_rulebooks.py` to create the vector stores.
5. Run `rpg_book_bot.py` to bring up the interface.