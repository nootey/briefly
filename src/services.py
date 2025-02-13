from googlesearch import search

def search_google(query):
    """
    Perform a Google search using the 'googlesearch' library and return structured results.
    """
    results = []
    try:
        for j in search(query, num_results=5, lang="en", sleep_interval=2, safe="active"):
            results.append({
                "title": j,  # Title is not available directly from `googlesearch`
                "link": j,  # URL of the result
                "snippet": f"Summary not available, visit {j} for details."  # Placeholder snippet
            })
    except Exception as e:
        print(f"Error during search: {e}")

    return {"organic_results": results}