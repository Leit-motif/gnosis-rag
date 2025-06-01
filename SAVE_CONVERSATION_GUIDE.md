# Obsidian Save Guide

Save your ChatGPT conversations directly to your Obsidian vault.

## The Solution

You now have a working **`save_conversation_save_conversation_post`** tool that saves exactly what you specify to today's daily note in your Obsidian vault.

## How to Use

### From ChatGPT or other tools:
Use the `save_conversation_save_conversation_post` tool with:

```json
{
  "conversation_name": "Your Conversation Title", 
  "content": "Your exact conversation content here"
}
```

### Example:
```json
{
  "conversation_name": "Python Debugging Help",
  "content": "**User:** My script is broken\n\n**Assistant:** I can help! What's the error?\n\n**User:** KeyError on line 5\n\n**Assistant:** Try using .get() instead of direct indexing."
}
```

## Where It Saves

- **Location**: `{vault_path}/{month}/{date}.md`  
- **Example**: `./data/01/2024-01-15.md`
- **Behavior**: Creates new file or appends to existing daily note

## Testing

Run this to verify everything works:
```bash
python test_save_conversation.py
```

## Troubleshooting

If saving fails:
1. **Check server is running** on the correct port
2. **Verify vault path** in `config.json` exists and is writable  
3. **Restart your integration** after making changes

That's it! Simple and working. 🎉 