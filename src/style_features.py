import numpy as np

def add_style_features(df):
    """
    Adds normalized style feature difference columns to df:
      - style_bold_count
      - style_header_count
      - style_list_count
      - style_sum_assistant_tokens
    Uses normdiff(a,b) = 0 if a+b==0 else (a-b)/(a+b)
    """
    def normdiff(a, b):
        denom = a + b
        return 0 if denom == 0 else (a - b) / denom

    style_bold, style_header, style_list, style_tokens = [], [], [], []

    for _, row in df.iterrows():
        meta = row["conv_metadata"]

        bold_a = meta.get("bold_count_a", {}).get("**", 0) + meta.get("bold_count_a", {}).get("__", 0)
        bold_b = meta.get("bold_count_b", {}).get("**", 0) + meta.get("bold_count_b", {}).get("__", 0)

        header_a = sum(meta.get("header_count_a", {}).values())
        header_b = sum(meta.get("header_count_b", {}).values())

        list_a = meta.get("list_count_a", {}).get("ordered", 0) + meta.get("list_count_a", {}).get("unordered", 0)
        list_b = meta.get("list_count_b", {}).get("ordered", 0) + meta.get("list_count_b", {}).get("unordered", 0)

        tok_a = meta.get("sum_assistant_a_tokens", 0)
        tok_b = meta.get("sum_assistant_b_tokens", 0)

        style_bold.append(normdiff(bold_a, bold_b))
        style_header.append(normdiff(header_a, header_b))
        style_list.append(normdiff(list_a, list_b))
        style_tokens.append(normdiff(tok_a, tok_b))

    df = df.copy()
    df["style_bold_count"] = style_bold
    df["style_header_count"] = style_header
    df["style_list_count"] = style_list
    df["style_sum_assistant_tokens"] = style_tokens
    return df
