import json
import requests
from bs4 import BeautifulSoup
from src import services
from src import knowledgebase as kb
from utils import py_helper as ph
import time

# Initialize Chroma Knowledge Base
chroma_client = kb.initialize_chroma_knowledge_base(db_path='knowledge_base')

def validate_company_name(openai_client, openai_model, extracted_company_name):

    print(f"     --> Validating company name: {extracted_company_name}")

    search_results = services.search_google(f"{extracted_company_name} company")

    search_snippets = "\n".join(
        [result["snippet"] for result in search_results.get("organic_results", [])[:5]]
    )

    prompt = f"""
    Based on the following search snippets, determine whether "{extracted_company_name}" 
    is a company or a product. If it is a product, identify the parent company it belongs to.

    **Search Snippets:**
    {search_snippets}

    **Instructions:**
    - Respond in JSON format:
      {{
        "is_product": true/false,
        "parent_company": "Name of the parent company" (if applicable)
      }}
    - If it's already a company, set "is_product" to false and "parent_company" to null.
    """

    response = openai_client.chat.completions.create(
        model=openai_model,
        temperature=0,
        messages=[
            {"role": "system",
             "content": "Determine if a term refers to a company or a product based on search results."},
            {"role": "user", "content": prompt}
        ]
    )

    validation_result = ph.handle_open_ai_json_extraction(response.choices[0].message.content)

    try:
        if validation_result["is_product"]:
            corrected_company_name = validation_result["parent_company"]
            print(f"     --> '{extracted_company_name}' is identified as a product of '{corrected_company_name}'.")
            return corrected_company_name
        else:
            print(f"     --> '{extracted_company_name}' is confirmed as a company.")
            return extracted_company_name
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from validation response: {e}")
        return extracted_company_name

def create_summary_l(openai_client, openai_model, transcript):

    prompt = f"""
    Create a detailed yet concise summary of the provided transcript. Keep in mind, that this will be used to inform colleagues of the same company,
    where the meeting is taking place. Optimize the summary, to be as helpful as reliable as possible, with the given context that you have.
    ONLY include the summary and key points, nothing else!

    **Important Instructions:**
    - Return the summary in plain text format.

    Transcript:
    {transcript}
    """

    response = openai_client.chat.completions.create(
        model=openai_model,
        temperature=0,
        messages=[
            {"role": "system", "content": "Extract structured business information from meeting transcripts."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def create_summary(openai_client, openai_model, transcript, summary_level="executive"):

    if summary_level == "executive":
        level_instruction = "Create a high-level executive summary focusing on strategic points and major takeaways."
    elif summary_level == "detailed":
        level_instruction = "Create a detailed summary with comprehensive insights, including technical details and operational specifics."
    else:
        level_instruction = "Create a summary of the transcript."

    prompt = f"""
    {level_instruction}

    **Meeting Transcript:**
    {transcript}

    **Important Instructions:**
    - Return the summary in plain text format.
    - Ensure the summary is concise and includes the most relevant points.
    """
    response = openai_client.chat.completions.create(
        model=openai_model,
        temperature=0,
        messages=[
            {"role": "system", "content": "Generate a meeting summary based on transcript."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def extract_keywords_from_transcript(openai_client, openai_model, transcript):

    prompt = f"""
    Extract the following from the company meeting transcript and return as a JSON object:

    - "company_name": Name of the company.
    - "keywords": {{
        "products": [Product names],
        "services": [Services],
        "market_areas": [Market areas]
    }}
    - "topics": {{
        "major_concerns": [Concerns],
        "opportunities": [Opportunities],
        "strategic_plans": [Plans]
    }}
    - "executives": [List of executives with roles, e.g., {{"name": "John Doe", "role": "CEO"}}]
    - "strategic_points": {{
        "financial_information": {{
            "product_pricing": {{"Product Name": "Price"}},
            "revenue_mentions": [],
            "cost_concerns": [],
            "budget_discussions": []
        }},
        "marketing_topics": {{
            "campaigns": [],
            "branding_strategies": [],
            "customer_engagement": [],
            "market_positioning": [],
            "competitors": []
        }}
    }}

    **Important Instructions:**
    - Return **only** the JSON object. No extra text.
    - Ensure all fields are present, even if empty (use empty lists or dictionaries).
    - Consistently format product pricing as a dictionary.

    Transcript:
    {transcript}
    """

    response = openai_client.chat.completions.create(
        model=openai_model,
        temperature=0,
        messages=[
            {"role": "system", "content": "Extract structured business information from meeting transcripts."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


def build_key_info_array(extracted_info):
    keywords = set()  # Deduplicate keywords automatically

    keywords_dict = extracted_info.get("keywords", {})
    strategic_points = extracted_info.get("strategic_points", {})
    topics = extracted_info.get("topics", {})

    for key, values in keywords_dict.items():
        if isinstance(values, list):
            keywords.update(values)
        elif isinstance(values, str):
            keywords.add(values)

    if isinstance(strategic_points, dict):
        pricing_info = strategic_points.get("pricing", {})
        if isinstance(pricing_info, dict):
            for product, price in pricing_info.items():
                product_name = product.replace('_', ' ')
                keywords.add(product_name)
                keywords.add(price)
    elif isinstance(strategic_points, list):
        for item in strategic_points:
            if isinstance(item, dict) and 'pricing' in item:
                pricing_info = item.get("pricing", {})
                for product, price in pricing_info.items():
                    product_name = product.replace('_', ' ')
                    keywords.add(product_name)
                    keywords.add(price)

    strategic_concepts = ['scalability', 'future-proofing', 'customer engagement']
    keywords.update(strategic_concepts)

    for topic_key, topic_value in topics.items():
        if isinstance(topic_value, list):
            keywords.update(topic_value)
        elif isinstance(topic_value, str):
            words = topic_value.split()
            for word in words:
                if len(word) > 4:
                    keywords.add(word.lower())

    filtered_keywords = [
        kw for kw in keywords
        if len(kw.strip()) > 2 and (len(kw.split()) <= 4)
    ]

    return filtered_keywords


def summarize_search_results(openai_client, openai_model, search_results):

    search_text = "\n".join([result["snippet"] for result in search_results.get("organic_results", [])[:5]])

    prompt = f"""
    Summarize the following information into key insights about the company:

    {search_text}

    Provide a concise summary that highlights recent developments, key business moves, and any relevant financial, strategic, or market updates.
    """

    response = openai_client.chat.completions.create(
        model=openai_model,
        temperature=0,
        messages=[{"role": "system", "content": "Summarize business insights based on search results."},
                  {"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def extract_and_summarize(openai_client, openai_model, urls):

    if not urls:
        return "No URLs found for this query."

    combined_text = ""
    for url in urls:
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract visible text
            page_text = ' '.join([p.get_text() for p in soup.find_all('p')])
            combined_text += page_text[:1000]  # Limit to avoid large payloads
        except Exception as e:
            print(f"Error fetching {url}: {e}")

    if combined_text:
        prompt = f"""
        Summarize the following information extracted from multiple sources about the topic:

        {combined_text}
        """

        response = openai_client.chat.completions.create(
            model=openai_model,
            temperature=0,
            messages=[
                {"role": "system", "content": "Summarize information about business topics."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content.strip()
    else:
        return "No relevant data found."


def generate_action_steps_l(openai_client, openai_model, transcript, company_summary):

    # Convert the aggregated summaries into a well-structured markdown format
    formatted_summary = "\n\n".join([f"### {keyword}\n{summary}" for keyword, summary in company_summary.items()])

    prompt = f"""
    Based on the following **corporate meeting transcript** and **company research summaries** (including financial and marketing insights), generate **personalized action steps** that align with the company's strategic goals.
    **Meeting Transcript:**
    {transcript}

    **Company Research Summary:**
    {formatted_summary}

    **Instructions:**
    - Focus on financial metrics such as product pricing, budget allocations, and revenue opportunities.
    - Highlight marketing initiatives, customer engagement strategies, and competitive positioning.
    - Identify key opportunities and challenges discussed in the transcript.
    - Relate them to the research summaries provided.
    - Focus on practical steps that improve customer satisfaction, address hardware limitations, enhance field measurements, and differentiate from competitors.
    - Make sure the action steps are **specific, relevant, and business-driven**.

    **Format the response in markdown format** with clear headings and bullet points.
    """

    response = openai_client.chat.completions.create(
        model=openai_model,
        temperature=0,
        messages=[{"role": "system", "content": "Generate strategic, actionable business steps."},
                  {"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def extract_kpis_and_timelines(openai_client, openai_model, transcript):

    prompt = f"""
    From the following meeting transcript, extract any Key Performance Indicators (KPIs) and any explicit deadlines or timelines mentioned.

    **Meeting Transcript:**
    {transcript}

    **Instructions:**
    - Identify and list any KPIs such as sales targets, product milestones, customer satisfaction scores, etc.
    - Identify any explicit deadlines or timeline references (e.g., "by next quarter", "within 2 weeks").
    - Return the results in JSON format with keys "kpis" (list) and "timelines" (list).
    - If none are found, return empty lists.
    """
    response = openai_client.chat.completions.create(
        model=openai_model,
        temperature=0,
        messages=[
            {"role": "system", "content": "Extract KPIs and timelines from meeting transcript."},
            {"role": "user", "content": prompt}
        ]
    )
    try:
        result = ph.handle_open_ai_json_extraction(response.choices[0].message.content)
    except Exception as e:
        print(f"Error extracting KPIs and timelines: {e}")
        result = {"kpis": [], "timelines": []}
    return result


def get_company_product_data(openai_client, openai_model, company_name, product_keywords):
    aggregated_product_data = {}
    for product in product_keywords:
        # Build a query that combines the company name and product keyword.
        query = f"{company_name} {product} product"
        print(f"     --> Searching for product data with query: {query}")
        search_results = services.search_google(query)
        # Extract URLs from the search results.
        urls = [result['link'] for result in search_results.get('organic_results', [])]
        if urls:
            summary = extract_and_summarize(openai_client, openai_model, urls)
            aggregated_product_data[product] = summary
        else:
            aggregated_product_data[product] = "No product data found."
    return aggregated_product_data


def generate_action_steps(openai_client, openai_model, transcript, company_summary, company_name, kpi_info=None,
                          previous_meeting_context=None, product_info=None):

    # Convert the aggregated summaries into a well-structured markdown format
    formatted_summary = "\n\n".join([f"### {keyword}\n{summary}" for keyword, summary in company_summary.items()])

    previous_context_text = ""
    if previous_meeting_context:
        previous_context_text = f"**Previous Meeting Context:**\n{previous_meeting_context}\n\n"

    # Format extracted KPIs and timelines (if any)
    extracted_kpis = kpi_info.get("kpis", []) if kpi_info else []
    extracted_timelines = kpi_info.get("timelines", []) if kpi_info else []
    metrics_text = ""
    if extracted_kpis:
        metrics_text += "KPIs: " + ", ".join(extracted_kpis) + "\n"
    if extracted_timelines:
        metrics_text += "Timelines: " + ", ".join(extracted_timelines) + "\n"

    # Format product data
    product_info_text = ""
    if product_info:
        product_info_text = "\n".join(
            [f"**Product URL:** {url}\nSummary: {info}" for url, info in product_info.items()])
        product_info_text = "**Company Product Data:**\n" + product_info_text + "\n\n"

    prompt = f"""
    Based on the following **corporate meeting transcript**, **company research summaries**, and any available **previous meeting context**, generate **personalized action steps** for the company:

    **Company name:**
    {company_name}

    **Meeting Transcript:**
    {transcript}

    **Company Research Summary:**
    {formatted_summary}

    {previous_context_text}

    **Extracted Metrics:**
    {metrics_text}

    {product_info_text}

    **Additional Instructions:**
    - Clearly separate objective details (e.g., pricing, hardware specifications) from subjective opinions (e.g., customer satisfaction).
    - Reference HBK’s actual product data where applicable.
    - Suggest novel ideas such as exploring alternative licensing models, design modifications for portability, or targeted market intelligence on competitor pricing.
    - The action steps should be specific, actionable, and aligned with HBK's strategic goals.
    - Format the response in markdown with clear headings and bullet points

    **Format the response in markdown format** with clear headings and bullet points.
    """
    response = openai_client.chat.completions.create(
        model=openai_model,
        temperature=0,
        messages=[
            {"role": "system", "content": "Generate strategic, actionable business steps."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def entrypoint_l(openai_client, transcript):

    collection_name = 'company_data_embed'
    openai_model = "gpt-4o"

    # kb.list_collection_data(chroma_client)
    # sys.exit()

    # kb.clear_chroma_collection(chroma_client, collection_name, True)
    # sys.exit()

    print("     --> Extracting key information from the transcript ...")
    extracted_info = extract_keywords_from_transcript(openai_client, openai_model, transcript)

    extracted_info = ph.handle_open_ai_json_extraction(extracted_info)

    # Extract and validate the company name
    extracted_company_name = extracted_info.get("company_name", "")
    validated_company_name = validate_company_name(openai_client, openai_model, extracted_company_name)
    validated_company_name = ph.sanitize_collection_name(validated_company_name)

    print("     --> Formatting extracted key information ...")
    formatted_info = build_key_info_array(extracted_info)

    print("     --> Searching for company data ...")
    aggregated_summaries = {}

    for key_info in formatted_info:
        print(f"     --> Retrieving from ChromaDB for {validated_company_name}: {key_info}")
        retrieved_docs = kb.retrieve_context_from_chroma(chroma_client, validated_company_name, key_info,
                                                         openai_client=openai_client)

        if retrieved_docs:
            print(f"     --> Found relevant data in ChromaDB for: {key_info}")
            aggregated_summaries[key_info] = " ".join(retrieved_docs)
        else:
            print(f"     --> No relevant data found in ChromaDB. Performing Google search for: {key_info}")

            search_results = services.search_google(f"{validated_company_name} {key_info}")
            urls = [result['link'] for result in search_results.get('organic_results', [])]

            print(f"     --> Extracting and summarizing data for {key_info} ...")
            summary = extract_and_summarize(openai_client, openai_model, urls)
            aggregated_summaries[key_info] = summary

            # Add to company-specific collection
            kb.add_to_chroma(chroma_client, validated_company_name, key_info, summary, openai_client, openai_model)

    print("     --> Generating action steps ...")
    action_steps = generate_action_steps(openai_client, openai_model, transcript, aggregated_summaries,
                                         company_name=validated_company_name)

    return action_steps


def entrypoint(openai_client, transcript):

    collection_name = 'company_data_embed'
    openai_model = "gpt-4o"

    print("     --> Extracting key information from the transcript ...")
    extracted_info = extract_keywords_from_transcript(openai_client, openai_model, transcript)
    extracted_info = ph.handle_open_ai_json_extraction(extracted_info)

    product_keywords = extracted_info.get("keywords", {}).get("products", [])

    # Validate and sanitize company name
    extracted_company_name = extracted_info.get("company_name", "")
    validated_company_name = validate_company_name(openai_client, openai_model, extracted_company_name)
    validated_company_name = ph.sanitize_collection_name(validated_company_name)

    print("     --> Formatting extracted key information ...")
    formatted_info = build_key_info_array(extracted_info)

    print("     --> Searching for company data ...")
    aggregated_summaries = {}
    for key_info in formatted_info:
        print(f"     --> Retrieving from ChromaDB for {validated_company_name}: {key_info}")
        retrieved_docs = kb.retrieve_context_from_chroma(chroma_client, validated_company_name, key_info,
                                                         openai_client=openai_client)
        if retrieved_docs:
            print(f"     --> Found relevant data in ChromaDB for: {key_info}")
            aggregated_summaries[key_info] = " ".join(retrieved_docs)
        else:
            print(f"     --> No relevant data found in ChromaDB. Performing Google search for: {key_info}")
            search_results = services.search_google(f"{validated_company_name} {key_info}")
            urls = [result['link'] for result in search_results.get('organic_results', [])]
            print(f"     --> Extracting and summarizing data for {key_info} ...")
            summary = extract_and_summarize(openai_client, openai_model, urls)
            aggregated_summaries[key_info] = summary
            # Add to company-specific collection
            kb.add_to_chroma(chroma_client, validated_company_name, key_info, summary, openai_client)

    # Extract KPIs and timeline information from the transcript
    print("     --> Extracting KPIs and timelines from transcript ...")
    kpi_info = extract_kpis_and_timelines(openai_client, openai_model, transcript)

    # Retrieve previous meeting context (if available)
    print("     --> Retrieving previous meeting context ...")
    previous_meeting_context = kb.retrieve_latest_meeting_summary(chroma_client, validated_company_name)

    # Retrieve company product data from the website
    print("     --> Retrieving company product data for additional context ...")
    product_info = get_company_product_data(openai_client, openai_model, validated_company_name, product_keywords)

    print("     --> Generating action steps ...")
    action_steps = generate_action_steps(
        openai_client,
        openai_model,
        transcript,
        aggregated_summaries,
        kpi_info=kpi_info,
        previous_meeting_context=previous_meeting_context,
        product_info=product_info,
        company_name=validated_company_name
    )

    # Generate current meeting summary (using executive level for brevity)
    print("     --> Generating current meeting summary ...")
    current_summary = create_summary(openai_client, openai_model, transcript, summary_level="executive")

    # Save the current meeting summary to the meeting history for continuous learning
    timestamp = int(time.time())
    kb.add_meeting_history(chroma_client, validated_company_name, current_summary, timestamp, openai_client)

    return action_steps