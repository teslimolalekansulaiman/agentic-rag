import os
import phoenix as px

# Unset the collector endpoint (this is the main issue)
if 'PHOENIX_COLLECTOR_ENDPOINT' in os.environ:
    del os.environ['PHOENIX_COLLECTOR_ENDPOINT']

# Close any phantom instances
px.close_app()

# Use environment variables instead of deprecated parameters
os.environ['PHOENIX_HOST'] = '127.0.0.1'
os.environ['PHOENIX_PORT'] = '6006'

# Launch fresh
session = px.launch_app()

print(f"Phoenix UI: {session.url}")

# Keep it running
input("Press Enter to stop Phoenix...")