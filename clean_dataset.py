import json
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_FILE = "data_extraction/alpaca_physics_5k_cleaned.jsonl"
OUTPUT_FILE = "data_extraction/alpaca_physics_5k_cleaned.jsonl" # Overwrite
BACKUP_FILE = "data_extraction/alpaca_physics_5k_cleaned.jsonl.bak"

def clean_text(text):
    """
    Attempts to parse text as JSON and extract meaningful content.
    If not JSON, returns original text.
    """
    cleaned = text.strip()
    # Quick check for JSON-like structure
    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            data = json.loads(cleaned)
            # Prioritize 'explanation' then 'answer'
            if "explanation" in data and data["explanation"]:
                return str(data["explanation"]).strip()
            elif "answer" in data and data["answer"]:
                return str(data["answer"]).strip()
            else:
                # If neither, join all values
                parts = [str(v).strip() for v in data.values() if v]
                return "\n".join(parts)
        except json.JSONDecodeError:
            pass
            
    return cleaned

def main():
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file {INPUT_FILE} not found.")
        return

    logger.info("Starting dataset cleaning...")
    
    # Create backup
    if os.path.exists(INPUT_FILE):
        import shutil
        shutil.copy2(INPUT_FILE, BACKUP_FILE)
        logger.info(f"Backup created at {BACKUP_FILE}")

    changed_count = 0
    total_count = 0
    new_lines = []

    with open(BACKUP_FILE, 'r') as f:
        for line in f:
            total_count += 1
            try:
                item = json.loads(line)
                original_output = item.get('output', '')
                cleaned_output = clean_text(original_output)
                
                if original_output != cleaned_output:
                    changed_count += 1
                    item['output'] = cleaned_output
                
                new_lines.append(json.dumps(item))
            except Exception as e:
                logger.warning(f"Skipping line due to error: {e}")
                new_lines.append(line) # Keep original if error

    # Write back
    with open(OUTPUT_FILE, 'w') as f:
        for line in new_lines:
            f.write(line + "\n")

    logger.info(f"Processing complete.")
    logger.info(f"Total lines: {total_count}")
    logger.info(f"Fixed lines: {changed_count}")

if __name__ == "__main__":
    main()
