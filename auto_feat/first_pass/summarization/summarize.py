import ast
import openai

MODEL = "argo:gpt-5"

client = openai.OpenAI(
    api_key="whatever+random",
    base_url="http://localhost:52483/v1",
)


def summarize(manuscript_path, data_path):

    """
    Reviews the manuscript and data, both in ASCII format, and provides a summary, a descriptive key
    for the data fields, and notes on both.

    Args:
        manuscript_path: path for the manuscript text file
        data_path: path for the data text file

    Returns:
        returns a dictionary with keys 'manuscript_summary', 'column_key', and 'notes'. 'column key' is a nested dictionary.

    """

    # Read the manuscript file
    try:
        with open(manuscript_path, 'r', encoding='utf-8') as f:
            manuscript_text = f.read()
    except Exception as e:
        print(f"Error reading manuscript file: {e}")
        return
    
    # Read the data file
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data_lines = f.readlines()[:5]  # Read first 5 lines
            data_text = ''.join(data_lines)
    except Exception as e:
        print(f"Error reading data file: {e}")
        return

    content = "Analyze the manuscript and data files provided in the text below " + \
        "and provide an output in the following nested dictionary format: \n" +\
        "{\n" +\
        " 'manuscript_summary':'<generated summary>',\n" +\
        " 'column key': {\n" +\
        "                '<column 1 name>': '<column 1 physical interpretation>',\n" +\
        "                '<column 2 name>': '<column 2 physical interpretation>',\n" +\
        "                ...\n" +\
        "                '<column M name>': '<column M physical interpretation>',\n" +\
        "               },\n" +\
        " 'notes: '<any notes or context that the LLM thinks is important to relay>'}\n\n" +\
        "manuscript text start\n---------------\n" +\
        manuscript_text +\
        "\n-------------------\nmanuscript text end\n\n" +\
        "data file start\n---------------\n" +\
        data_text +\
        "\n-------------------\ndata file end\n\n"

    messages = [
        {
            "role": "user",
            "content": content,
        },
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        ).choices[0].message.content.strip()
    except Exception as e:
        print("\nError:", e)

    # review and correct the previously generated content
    content_review = "Review the LLM response included below and make any necessary corrections " +\
        "to the syntax, format and content. The response should be in the form of a nested dictionary with " +\
        "top level keys 'manuscript_summary', 'column_key', and 'notes'. 'column key' is a dictionary." +\
        "Also consider the original prompt that was provided, as included below.\n\n" +\
        "input_string start\n-----\n" +\
        response +\
        "\n-----\ninput_string end" + \
        "previous prompt start\n-----\n" +\
        content +\
        "\n-----\nprevious prompt end"

    messages = [
        {
            "role": "user",
            "content": content_review,
        },
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        ).choices[0].message.content.strip()
        return ast.literal_eval(response)
    except Exception as e:
        print("\nError:", e)


if __name__ == "__main__":

    data_path = "data.csv"
    manuscript_path = "manuscript.txt"

    response = summarize(manuscript_path, data_path)

    print("manuscript summary: ", response['manuscript_summary'], "\n\n")
    print("column key: (first entry only)", response['column_key']['PROPERTY: Exp. Density (g/cm$^3$)'], "\n\n")
    print("notes: ", response['notes'], "\n\n")
