import streamlit as st
import json
import os
import re
from mistralai import Mistral
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from typing import List, Dict, Any

# CPC class descriptions for display
CPC_DESCRIPTIONS = {
    "A01": "Agriculture; forestry; animal husbandry; hunting; trapping; fishing",
    "A47": "Furniture; domestic articles; coffee mills; spice mills; suction cleaners in general",
    "A61": "Medical or veterinary science; hygiene",
    "A63": "Sports; games; amusements",
    "B01": "Physical or chemical processes or apparatus in general",
    "B23": "Machine tools; metal-working not otherwise provided for",
    "B25": "Hand tools; portable power-driven tools; manipulators",
    "B29": "Working of plastics; working of substances in a plastic state in general",
    "B32": "Layered products",
    "B33": "Additive manufacturing technology",
    "B41": "Printing; lining machines; typewriters; stamps",
    "B60": "Vehicles in general",
    "B62": "Land vehicles for travelling otherwise than on rails",
    "B64": "Aircraft; aviation; cosmonautics",
    "B65": "Conveying; packing; storing; handling thin or filamentary material",
    "C01": "Inorganic chemistry",
    "C07": "Organic chemistry",
    "C08": "Organic macromolecular compounds; their preparation or chemical working-up; compositions based thereon",
    "C09": "Dyes; paints; polishes; natural resins; adhesives; compositions not otherwise provided for; applications of materials not otherwise provided for",
    "C12": "Biochemistry; beer; spirits; wine; vinegar; microbiology; enzymology; mutation or genetic engineering",
    "C23": "Coating metallic material; coating material with metallic material; chemical surface treatment; diffusion treatment of metallic material; coating by vacuum evaporation, by sputtering, by ion implantation or by chemical vapour deposition, in general; inhibiting corrosion of metallic material or incrusted in general",
    "E21": "Earth or rock drilling; mining",
    "F01": "Machines or engines in general; engine plants in general; steam engines",
    "F02": "Combustion engines; hot-gas or combustion-product engine plants",
    "F04": "Positive-displacement machines for liquids; pumps for liquids or elastic fluids",
    "F05": "Indexing schemes relating to engines or pumps in various subclasses of classes F01-F04",
    "F16": "Engineering elements and units; general measures for producing and maintaining effective functioning of machines or installations; thermal insulation in general",
    "F21": "Lighting",
    "F24": "Heating; ranges; ventilating",
    "G01": "Measuring; testing",
    "G02": "Optics",
    "G03": "Photography; cinematography; analogous techniques using waves other than optical waves; electrography; holography",
    "G05": "Controlling; regulating",
    "G06": "Computing; calculating or counting",
    "G07": "Checking-devices",
    "G08": "Signalling",
    "G09": "Education; cryptography; display; advertising; seals",
    "G10": "Musical instruments; acoustics",
    "G11": "Information storage",
    "G16": "Information and communication technology [ICT] specially adapted for specific application fields",
    "H01": "Electric elements",
    "H02": "Generation, conversion, or distribution of electric power",
    "H03": "Electronic circuitry",
    "H04": "Electric communication technique",
    "H05": "Electric techniques not otherwise provided for",
    "H10": "Semiconductor devices; electric solid-state devices not otherwise provided for",
    "Y02": "Technologies or applications for mitigation or adaptation against climate change",
    "Y10": "Technical subjects covered by former USPC"
}

def count_tokens(prompt: str) -> int:
    """Count tokens in a prompt using Mistral tokenizer."""
    try:
        tokenizer = MistralTokenizer.v3(is_tekken=True)
        model_name = "ministral-8b-2410"
        tokenizer = MistralTokenizer.from_model(model_name)

        # Tokenize a list of messages
        tokenized = tokenizer.encode_chat_completion(
            ChatCompletionRequest(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
            )
        )
        tokens = tokenized.tokens
        return len(tokens)
    except Exception:
        # Fallback: rough estimate of tokens
        return len(prompt.split()) * 1.3

def remove_tables(text: str) -> str:
    """Remove tables from patent description."""
    # Matches 'TABLE' (all caps), followed by any characters (non-greedy), up to a double newline or end of string.
    pattern = re.compile(r"TABLE[\s\S]+?(?=(\n\n|$))")
    return re.sub(pattern, "", text)

def process_description(desc: str) -> str:
    """Process patent description with truncation and cleaning."""
    words = desc.split(" ")
    if len(words) < 5000:
        return desc

    # Truncate to 5000 words and add ellipsis
    truncated_words = words[:5000]
    desc = " ".join(truncated_words) + " ..."

    # Check token count
    token_count = count_tokens(desc)
    if token_count <= 10000:
        return desc

    # Remove tables if still too long
    desc_no_tables = remove_tables(desc)
    if count_tokens(desc_no_tables) <= 10000:
        return desc_no_tables

    # Remove words longer than 50 characters as a last resort
    filtered_words = [word for word in truncated_words if len(word) <= 50]
    desc = " ".join(filtered_words)
    return desc

def initialize_mistral_client():
    """Initialize Mistral client with API key."""
    api_key = os.environ.get("MISTRAL")
    if not api_key:
        st.error("MISTRAL API key not found. Please set the MISTRAL environment variable.")
        return None
    return Mistral(api_key=api_key)

def classify_patent_text(client: Mistral, model_name: str, patent_text: str) -> list:
    """
    Classify patent text using either base model with prompts or fine-tuned model.
    
    Args:
        client: Mistral client instance
        model_name: Name of the model (base or fine-tuned)
        patent_text: Patent description text
        
    Returns:
        List of predicted CPC class IDs
    """
    try:
        # Process patent text first
        processed_text = process_description(patent_text)
        
        # Check if it's a fine-tuned model
        if model_name.startswith("ft:"):
            # Use classifier API for fine-tuned models
            classifier_response = client.classifiers.classify(
                model=model_name,
                inputs=[processed_text],
            )
            
            # Extract class IDs from classifier response
            result = classifier_response.model_dump()
            class_ids = []
            
            if 'results' in result and result['results']:
                classification = result['results'][0]
                if 'cpc_class_ids' in classification:
                    scores = classification['cpc_class_ids']["scores"]
                    class_ids.extend([k for k, v in scores.items() if v > 0.05])
            return class_ids
        else:
            # Use chat completion with prompts for base models
            sys_prompt = open("prompts/sys_prompt.md", "r").read()
            user_prompt = open("prompts/user_prompt.md", "r").read()
            
            response = client.chat.complete(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": user_prompt.format(patent_description=processed_text),
                    },
                ],
                temperature=0,
            )
            
            response_content = response.choices[0].message.content
            
            # Clean up response content (remove markdown formatting)
            response_content = response_content.replace("```json\n", "").replace("\n```", "")
            
            # Convert response to JSON array
            try:
                return json.loads(response_content)
            except json.JSONDecodeError:
                st.warning(f"Invalid JSON response: {response_content}")
                return []
                
    except Exception as e:
        st.error(f"Error during classification: {str(e)}")
        return []

def main():
    st.set_page_config(
        page_title="Patent CPC Classifier",
        page_icon="📋",
        layout="wide"
    )
    
    st.title("🔬 Patent CPC Classification Tool")
    st.markdown("Enter a patent description to predict relevant Cooperative Patent Classification (CPC) codes.")
    
    # Initialize Mistral client
    client = initialize_mistral_client()
    if not client:
        st.stop()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model selection dropdown
    model_options = [
        "ministral-3b-latest",
        "ministral-8b-latest", 
        "mistral-small-latest",
        "mistral-medium-latest"
    ]
    
    model_name = st.sidebar.selectbox(
        "Select Model",
        options=model_options,
        index=0,
        help="Choose a Mistral model for patent classification"
    )
    
    # Option to use fine-tuned model
    use_finetuned = st.sidebar.checkbox("Use Fine-tuned Model")
    
    if use_finetuned:
        ft_suffix = st.sidebar.text_input(
            "Fine-tuned Model Suffix",
            value="",
            help="Enter your fine-tuned model suffix (e.g., 'ds1')"
        )
        if ft_suffix:
            model_name = f"ft:{model_name}:{ft_suffix}"
    
    # Main input area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Patent Description")
        patent_text = st.text_area(
            "Enter patent description:",
            height=300,
            placeholder="Describe the invention, its technical features, and functionality...",
            help="Enter the patent text you want to classify. The model will predict relevant CPC codes."
        )
        
        classify_button = st.button("🔍 Classify Patent", type="primary")
    
    with col2:
        st.header("About CPC Codes")
        st.markdown("""
        **Cooperative Patent Classification (CPC)** is an international patent classification system.
        
        **Main Sections:**
        - **A**: Human Necessities
        - **B**: Operations; Transporting
        - **C**: Chemistry; Metallurgy
        - **E**: Fixed Constructions
        - **F**: Mechanical Engineering
        - **G**: Physics
        - **H**: Electricity
        - **Y**: Emerging Technologies
        """)
    
    # Classification results
    if classify_button:
        if not patent_text.strip():
            st.error("Please enter a patent description before classifying.")
        else:
            with st.spinner("Classifying patent text..."):
                classification_result = classify_patent_text(client, model_name, patent_text.strip())
                
                if classification_result:
                    st.success("Classification completed!")
                    
                    # Display results
                    st.header("🎯 Predicted CPC Codes")
                    
                    # Create formatted results with descriptions
                    formatted_results = []
                    for class_id in classification_result:
                        description = CPC_DESCRIPTIONS.get(class_id, "Description not available")
                        formatted_results.append({
                            'CPC Code': class_id,
                            'Description': description
                        })
                    
                    if formatted_results:
                        import pandas as pd
                        df = pd.DataFrame(formatted_results)
                        st.dataframe(df, use_container_width=True)
                        
                        # Show raw response (collapsible)
                        with st.expander("Raw Classification Response"):
                            st.json(classification_result)
                    else:
                        st.warning("No classification results found.")
                else:
                    st.error("Classification failed. Please check your model name and try again.")
    

if __name__ == "__main__":
    main()