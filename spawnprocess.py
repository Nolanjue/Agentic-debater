
import subprocess
import requests


def run_node_script(query):
    try:
        result = subprocess.run(['node', 'scraper.js', query], capture_output=True, text=True,  encoding='utf-8')
        print("Node.js script output:")
        print(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing Node.js script: {e}")

# Example usage

query = "global warming is real"
value = run_node_script(query)

print(value)

