import openai
import os
import re

openai.api_key = 'YOUR_OPENAI_API_KEY'

def process_file(common_path, file_name):
    directory = os.path.join(common_path, file_name)
    print(f'The sum of weighted_indices count/max/Perplexity/Corrupted')
    with open(directory, 'r') as file:
        content = file.readlines()

    # Initialize regex patterns
    generation_pattern = re.compile(r'(\d+)-th generation')  # Matches A-th generation
    sum_pattern = re.compile(r'The sum of weighted_indices: ([\d.]+)')  # Matches YYY
    perplexity_pattern = re.compile(r'Perplexity: ([\d.]+)')  # Matches ZZZ
    ms_pattern = re.compile(r"The number of 'MS' tokens: (\d+)")  # Matches the number of 'MS' tokens

    # Initialize tracking variables
    max_sum = 0  # To track the maximum sum for the current generation
    count_sum = 0  # To count occurrences of 'The sum of weighted_indices'
    current_generation = None
    perplexity = None
    corrupted = None
    example_num = 0
    corrupted_sum = 0

    for line in content:
        # Check for 'A-th generation'
        generation_match = generation_pattern.search(line)
        if generation_match:
            # Print results of the previous generation if any
            if current_generation is not None:
                print(f"{count_sum} {max_sum} {perplexity} {corrupted}")

            # Start a new generation
            current_generation = int(generation_match.group(1))
            max_sum = 0  # Reset for the new generation
            count_sum = 0
            perplexity = None
            corrupted = None

        # Check for 'The sum of weighted_indices'
        sum_match = sum_pattern.search(line)
        if sum_match:
            current_sum = float(sum_match.group(1))
            count_sum += 1  # Increment the count
            if current_sum > max_sum:
                max_sum = current_sum  # Update the maximum sum

        # Check for 'Perplexity'
        perplexity_match = perplexity_pattern.search(line)
        if perplexity_match:
            perplexity = float(perplexity_match.group(1))

        # Check for "The number of 'MS' tokens"
        ms_match = ms_pattern.search(line)
        if ms_match:
            num_ms_tokens = int(ms_match.group(1))
            corrupted = 1 if num_ms_tokens >= 3 else 0
            corrupted_sum += corrupted
            example_num += 1

    # Print the result for the last generation
    if current_generation is not None:
        print(f"{count_sum} {max_sum} {perplexity} {corrupted}")
    print(f'corrupted_sum: {corrupted_sum} / example_num: {example_num}')
    print(f'Corrupted Text Ratio: {corrupted_sum/example_num*100:.02f}%')
    
def make_template(i, text_content):
    template = f"""{text_content}"""
    return template

def process_response(
    common_path, file_name
):
    # file_name = 'Please introduce yourself._seed42.txt'
    read_path = os.path.join(common_path, file_name)
    write_path = os.path.join(read_path + '_responses')
    os.makedirs(write_path, exist_ok=True)
    # output_path = common_path + 'Please_introduce_yourself._score.txt'
    with open(read_path, "r", encoding="utf-8") as f_in:
        lines = [line.rstrip("\n") for line in f_in]
    
    results = []
    
    i = 0
    collecting = False
    current_text = []
    
    for line in lines:
        if "Generated response with" in line:
            if current_text:
                i += 1
                results.append(make_template(i, "\n".join(current_text)))
                current_text = []
            
            collecting = True
            continue
        
        if "Perplexity:" in line:
            if collecting:
                i += 1
                results.append(make_template(i, "\n".join(current_text)))
                current_text = []
                collecting = False
            continue
        
        if collecting:
            current_text.append(line)
    
    if current_text:
        i += 1
        results.append(make_template(i, "\n".join(current_text)))
    
    for i in range(1, len(results)+1):
        response_path = os.path.join(write_path, f'response_{i}.txt')
        with open(response_path, 'w', encoding='utf-8') as f:
            f.write(results[i-1])
    
    print(f">>>{i} blocks were Processed.총 {i}")

def extract_scores(output_file_path, model_name):
    """
    Extracts and prints all occurrences of 'X' from lines containing 'X points: ...'
    based on the specific spacing pattern provided.
    """
    print(f'output_file_path: {output_file_path}')
    
    scores = []

    with open(output_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        if lines[i].strip() == f"{model_name} Response:":
            if i + 2 < len(lines):
                line_below = lines[i + 2].strip()
                parts = line_below.split()
                if len(parts) > 0:
                    part = parts[0].lstrip('*')
                    if part.isdigit():
                        scores.append(int(part))
                        print(part)
                    else:
                        print('XXX')
                    

    print("Extracted Scores:", len(scores))
    return scores

def ask_gpt4(criteria, response):
    messages = [
        {"role": "system", "content": criteria},
        {"role": "user", "content": response}
    ]
    
    response_from_gpt = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=2000,
        temperature=0
    )
    
    return response_from_gpt.choices[0].message.content.strip()