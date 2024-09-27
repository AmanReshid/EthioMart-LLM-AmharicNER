import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class TelegramDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.model = None
        self.tokenizer = None
        self.categories = {
            'kids': ['toy', 'children', 'kids', 'መጫወቻ', 'play', 'games', 'fun', 'educational', 'puzzle', 'doll', 'action figure', 'stuffed animal', 'arts and crafts', 'books', 'outdoor toys', 'building blocks', 'baby', 'toddler', 'Baby','መጫወቻዎች'],
            'men': ['men', 'grooming', 'shaving', 'beard', 'razor', 'aftershave', 
                    'scent', 'deodorant', 'grooming kit', 'haircut', 'fashion', 'suits', 
                    'wallet', 'watch', 'accessories', 'fitness', 'shoes', 
                    'አስተካክል', 'የብርሃን ዕቃዎች'],
            'women': ['women', 'makeup', 'hair dryer', 'lipstick', 'foundation', 'mascara', 
                        'skincare', 'nails', 'jewelry', 'dresses', 'handbags', 'accessories', 
                        'fashion', 'shoes', 'perfume', 'hairstyle', 'wellness', 'beauty', 'style','Hair Drye',
                        'እንቅስቃሴ', 'የፀጉር እቃዎች', 'የውበት እቃዎች'],
            'sport': ['gym', 'GYM','fitness', 'exercise', 'እንቅስቃሴ', 'workout', 'training', 'yoga', 
                        'running', 'cycling', 'sportswear', 'equipment', 'weights', 'cardio', 
                        'aerobics', 'team sports', 'outdoor activities', 'athletics', 'health',  'workout', 'sports',
                        'ስፖርት', 'የእንቅስቃሴ መሳሪያዎች'],
            'groceries': ['food', 'snacks', 'grocery', 'ምግብ', 'produce', 'fruits', 'vegetables', 
                            'meat', 'dairy', 'bread', 'cereal', 'beverages', 'frozen', 'canned', 
                            'organic', 'bulk', 'condiments', 'spices', 'snack bars', 'breakfast', 
                            'እንቁላል', 'ወተር', 'የምግብ እቃዎች'],
             'accessories': ['jewelry', 'bags', 'accessory', 'ቀለበት', 'belts', 'hats', 'scarves', 
                        'sunglasses', 'watches', 'hair accessories', 'wallets', 'phone cases', 
                        'keychains', 'pins', 'brooches', 'fashion', 'style', 'gifts', 'decor', 'የልብስ መቶከሻ\n\n',
                        'የመልክዕ እቃዎች', 'የምታወቅ እቃዎች','Anti-theft ',' Earbuds','PowerBank','Grip Tape','humidifier'],
            'health': ['health', 'ጤና', 'wellness', 'nutrition', 'vitamins', 'supplements', 
                        'exercise', 'fitness', 'mental health', 'meditation', 'stress relief', 
                        'doctor', 'check-up', 'first aid', 'hygiene', 'immune system', 'balance', 
                        'self-care', 'አንደኛ ጤና', 'የጤና እቃዎች','pulse'],
             'household': ['cleaning', 'furniture', 'decor', 'appliances', 'utensils', 'kitchen', 
                        'bathroom', 'laundry', 'storage', 'organization', 'home improvement', 'pan', 
                        'gardening', 'tools', 'supplies', 'safety', 'maintenance', 'pets', 'spatulas','Kitchen','Mop',
                        'spatulas\n\n','nለኪችንዎ','home', 'comfort', 'ቤት', 'የቤት እቃዎች', 'እንቅስቃሴ','bottle','ፔርሙስ','knife',
                        'Glass','የላዛኛ','stove','Ironing Board','Slicer','BLENDER','MULTIFUNCTIONAL BLENDER','Toilet Brush',
                        'የቢላ ስብስብ','ቢላ','Oven','fridge', 'መጥበሻ','Toilet','Mob','cookware','Blender','KITCHENWARE','ምንጣፍ','Tablemats']
        }

    def check_and_remove_nan(self, column_name):
        print(f"Checking for NaN values in the '{column_name}' column:")
        nan_count = self.df[column_name].isnull().sum()
        print(f"Number of NaN values in '{column_name}' column: {nan_count}")
        self.df = self.df.dropna(subset=[column_name])
        print(f"Dataset shape after dropping NaN values in '{column_name}' column: {self.df.shape}")

    def remove_emojis(self, text):
        """Removes emojis including '1️⃣' from text."""
        emoji_pattern = re.compile(
            "[" 
            "\U0001F600-\U0001F64F"  
            "\U0001F300-\U0001F5FF" 
            "\U0001F680-\U0001F6FF"  
            "\U0001F700-\U0001F77F"  
            "\U0001F1E0-\U0001F1FF"  
            "\u0031\uFE0F\u20E3"     
            "]+", 
            flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)

    def clean_messages(self):
        self.df['Message'] = self.df['Message'].apply(self.remove_emojis)
        self.df.to_csv('../data/clean_data.csv', index=False)
        print("Cleaned data saved.")

    def label_message(self, message):
        """Labels messages with prices and locations using a rule-based approach."""
        # Define multi-word entities (locations, products, etc.)
        multi_word_entities = {
            'ብስራተ ገብርኤል': 'I-LOC',
        }

        # First, check for multi-word entities in the message
        for entity, label in multi_word_entities.items():
            if entity in message:
                message = message.replace(entity, f"{entity.replace(' ', '_')}")  # Replace spaces with underscores

        tokens = re.findall(r'\S+', message)  # Tokenize after replacing multi-word entities
        labeled_tokens = []

        for token in tokens:
            # After tokenizing, replace underscores with spaces again for multi-word entities
            token = token.replace('_', ' ')

            # Check if token is a multi-word entity (location, product, etc.)
            if token in multi_word_entities:
                labeled_tokens.append(f"{token} {multi_word_entities[token]}")
            # Check if token is a location (single-word locations)
            elif any(loc in token for loc in ['ገርጂ', '4ኪሎ']):
                labeled_tokens.append(f"{token} I-LOC")
            # Check if token is a phone number (exclude numbers longer than 9 digits)
            elif re.match(r'^\+?\d{10,15}$', token):
                labeled_tokens.append(f"{token} O")
            # Check if token is a price (e.g., 500 ETB, $100, or ብር)
            elif re.match(r'^\d+(\.\d{1,2})?$', token) and len(token) < 9:
                labeled_tokens.append(f"{token} I-PRICE")
            elif 'ብር' in token or 'Birr' in token or 'ETB' in token:
                labeled_tokens.append(f"{token} I-PRICE")
            # Otherwise, treat it as outside any entity
            else:
                labeled_tokens.append(f"{token} O")

        return "\n".join(labeled_tokens)

    def apply_labeling(self):
        self.df['Labeled_Message'] = self.df['Message'].apply(self.label_message)
        labeled_data_path = '../data/labeled_telegram_data.txt'
        with open(labeled_data_path, 'w', encoding='utf-8') as f:
            for _, row in self.df.iterrows():
                f.write(f"{row['Labeled_Message']}\n\n")
        print(f"Labeled data saved to {labeled_data_path}")
    


    def load_ner_model(self):
        # Loads a pretrained NER model with error handling
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0")
            self.model = AutoModelForTokenClassification.from_pretrained("masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0")
            self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
            print("NER model loaded successfully.")
        except Exception as e:
            print(f"Error loading the NER model: {e}")


    def apply_ner(self):
        # Applies NER to the 'Message' column using a pretrained model.
        if self.model is None or self.tokenizer is None:
            self.load_ner_model()

        # Ensure the 'Message' column exists in the DataFrame
        if 'Message' not in self.df.columns:
            print("Error: 'Message' column not found in the DataFrame.")
            return

        # Apply NER
        try:
            ner_results = self.nlp(self.df['Message'].tolist())
            print("NER results:", ner_results)
            return ner_results  # You can return the results for further processing
        except Exception as e:
            print(f"Error applying NER: {e}")

    def is_amharic(self, message):
        # Checks if a string contains Amharic characters.
        return bool(re.search(r'[\u1200-\u137F]', message))

    def classify_message(self, message):
        # Classifies messages based on predefined categories.
        if pd.isna(message):
            return 'uncategorized'

        if self.is_amharic(message):
            for category, keywords in self.categories.items():
                if any(keyword in message for keyword in keywords):
                    return category
        else:
            for category, keywords in self.categories.items():
                if any(keyword in message.lower() for keyword in keywords):
                    return category
        return 'uncategorized'

    def apply_classification(self):
        # Applies the classification to the 'Message' column.
        self.df['Category'] = self.df['Message'].apply(self.classify_message)
        print(self.df[['Message', 'Category']])

    def save_classified_data(self):
        # Displays category counts and saves uncategorized messages.
        category_counts = self.df['Category'].value_counts()
        print(category_counts)
        uncategorized_items = self.df[self.df['Category'] == 'uncategorized']
        uncategorized_items.to_csv('../data/uncategorized_data.csv', index=False)
        self.df.to_csv('../data/labeled_data.csv', index=False)
        print("Labeled and uncategorized data saved.")