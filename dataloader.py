from sklearn.model_selection import train_test_split
import pandas as pd
import re


def load_csv(file: str = "./data/training.1600000.processed.noemoticon.csv"):
    csv_file = pd.read_csv(file, encoding="latin1", header=None)
    csv_file.columns = ['polarity', 'id', 'date', 'query', 'user', 'text']

    data = csv_file[["polarity", "text"]]

    data['polarity'] = data['polarity'].replace({4: 1, 0: 0})

    return data

def clear_data(text: str):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    return text.lower()

def create_split():
    data = load_csv()
    data["text"] = data["text"].apply(clear_data)

    X = data["text"]
    y = data["polarity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    
    print(f"TRAIN SIZE: {len(X_train)}")
    print(f"TEST SIZE: {len(X_test)}")

    return (X_train, X_test, y_train, y_test)

def save_split(X_train, X_test, y_train, y_test):
    train_data = pd.DataFrame({"text": X_train, "polarity":y_train})
    test_data = pd.DataFrame({"text": X_test, "polarity":y_test})

    train_data.to_csv('./data/train_sentiment140.csv', index=False)
    test_data.to_csv('./data/test_sentiment140.csv', index=False)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = create_split()
    save_split(X_train, X_test, y_train, y_test)

