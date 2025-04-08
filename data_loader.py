import pandas as pd
import re
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Define the common pronouns to identify
PRONOUNS = ['he', 'she', 'they', 'it', 'I', 'we', 'you', 'one', 'who', 'that', 'which']

def load_data_from_csv(file_path):
    """
    Load data from a CSV file containing texts and optionally labels
    
    Expected format:
    - text: column containing the text
    - label: (optional) column containing the pronoun label
    """
    df = pd.read_csv(file_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist() if 'label' in df.columns else None
    
    return texts, labels

def load_data_from_txt(file_path, delimiter='\t'):
    """
    Load data from a text file where each line contains text and optionally a label
    
    Expected format:
    - text<delimiter>label (if labeled)
    - text (if unlabeled)
    """
    texts = []
    labels = []
    has_labels = False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if delimiter in line:
                parts = line.split(delimiter, 1)  # Split only on the first delimiter
                text = parts[0].strip()
                label = parts[1].strip()
                texts.append(text)
                labels.append(label)
                has_labels = True
            else:
                texts.append(line)
    
    return texts, labels if has_labels else None

def extract_pronouns_from_text(text):
    """
    Extract all pronouns from a text and return counts
    """
    text_lower = text.lower()
    pronoun_counts = {}
    
    # Count occurrences of each pronoun
    for pronoun in PRONOUNS:
        # Use regex to find whole word matches only
        pattern = r'\b' + pronoun + r'\b'
        count = len(re.findall(pattern, text_lower))
        if count > 0:
            pronoun_counts[pronoun] = count
    
    return pronoun_counts

def find_primary_pronoun(text):
    """
    Find the most commonly used pronoun in a text
    """
    pronoun_counts = extract_pronouns_from_text(text)
    
    if not pronoun_counts:
        return "unknown"  # No pronouns found
    
    # Find the pronoun with the highest count
    primary_pronoun = max(pronoun_counts.items(), key=lambda x: x[1])[0]
    return primary_pronoun

def create_pronoun_mappings(texts=None, labels=None):
    """
    Create mappings between pronouns and indices
    
    If labels are provided, uses them to create the mapping
    If only texts are provided, extracts pronouns from texts
    """
    unique_pronouns = set()
    
    if labels is not None:
        # Use the provided labels
        unique_pronouns = set(labels)
    elif texts is not None:
        # Extract pronouns from texts
        for text in texts:
            pronoun_counts = extract_pronouns_from_text(text)
            if pronoun_counts:
                primary_pronoun = max(pronoun_counts.items(), key=lambda x: x[1])[0]
                unique_pronouns.add(primary_pronoun)
    
    # Add common pronouns if not already in the set
    for pronoun in PRONOUNS:
        unique_pronouns.add(pronoun)
    
    # Add "unknown" for cases with no pronouns
    unique_pronouns.add("unknown")
    
    # Sort for reproducibility
    sorted_pronouns = sorted(list(unique_pronouns))
    
    # Create mappings
    pronoun_to_idx = {pronoun: i for i, pronoun in enumerate(sorted_pronouns)}
    idx_to_pronoun = {i: pronoun for i, pronoun in enumerate(sorted_pronouns)}
    
    return pronoun_to_idx, idx_to_pronoun

class PronounDataset(Dataset):
    def __init__(self, texts, tokenizer, pronoun_to_idx=None, is_training=True, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.pronoun_to_idx = pronoun_to_idx
        self.is_training = is_training
        self.max_length = max_length
        
        # Preprocess data
        if is_training and pronoun_to_idx is not None:
            self.labels = []
            for text in texts:
                primary_pronoun = find_primary_pronoun(text)
                # Use the "unknown" class if the pronoun is not in our mapping
                label_idx = pronoun_to_idx.get(primary_pronoun, pronoun_to_idx["unknown"])
                self.labels.append(label_idx)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Convert to tensors
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        token_type_ids = encoding['token_type_ids'].squeeze()
        
        if self.is_training:
            label = self.labels[idx]
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }

def prepare_dataset(texts, tokenizer, pronoun_to_idx=None, is_training=True, max_length=512):
    """
    Prepare a dataset for training or inference
    """
    return PronounDataset(texts, tokenizer, pronoun_to_idx, is_training, max_length)

def prepare_train_val_test_split(texts, test_size=0.1, val_size=0.1, random_state=42):
    """
    Split data into train, validation, and test sets
    """
    # First split: separate out test set
    train_val_texts, test_texts = train_test_split(
        texts, test_size=test_size, random_state=random_state
    )
    
    # Second split: split remaining data into train and validation
    # Calculate the validation size as a proportion of the remaining data
    val_proportion = val_size / (1 - test_size)
    train_texts, val_texts = train_test_split(
        train_val_texts, test_size=val_proportion, random_state=random_state
    )
    
    return train_texts, val_texts, test_texts

# For testing
if __name__ == "__main__":
    # Example text
    example_text = "This is what it takes to get into Harvard University. Here's a college application of a girl who did and into Harvard. She had a 4.1 GPA, a 1560 SAT score, and a 36, a perfect score on the AC. Comes from a high income family, her dad went to Harvard and her mom went to UCLA. He intense the major in business, his valedictorian, and she only had two extracurriculars in school. She did varsity track and varsity cross country. On top of that, she also had a business that made $30,000 in profit last year. So something that she started, her sophomore year. And as for her college results, well, she was accepted to Georgetown. Ah, I'm sorry. She actually"