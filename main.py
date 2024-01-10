import json
import shutil
import sys
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

# Inițializează lematizatorul
# lemmatizator = WordNetLemmatizer()
#
# # Exemple de lematizare
# cuvant1 = lemmatizator.lemmatize("running", pos='v')  # 'pos' specifică partea de vorbire
# cuvant2 = lemmatizator.lemmatize("better", pos='a')
# cuv3 = lemmatizator.lemmatize("ve", pos='v')
# part_prop = pos_tag(["ve"])
# print(part_prop)
# #cuv3 = lemmatizator.lemmatize("got", pos=part_prop[0][1])
#
# print(cuvant1)  # Output: run
# print(cuvant2)  # Output: good
# print(cuv3)


# text = "NLTK is a powerful library for natural language processing."
# words = word_tokenize(text)
# sentences = sent_tokenize(text)
#
# print(words)
# print(sentences)
#
# stop_words = set(stopwords.words("english"))
# filtered_words = [word for word in words if word.lower() not in stop_words]
#
# print(filtered_words)
# print(stop_words)

# walk through the files and filter the stop words



# path = path to bare, lemn, lemn_stop or stop
def reset_folder(folder):
    try:
        for item in os.listdir(folder):
            if os.path.isfile(os.path.join(folder, item)):
                os.remove(os.path.join(folder, item))
            else:
                item_path = os.path.join(folder, item)
                shutil.rmtree(item_path)

    except Exception as e:
        if len(os.listdir(folder)) != 0:
            print("Error: Could not revert the folder", file=e)
        else:
            print("Something has gone terribly wrong", file=sys.stderr)
        raise SystemExit

def adjust_pos_tag(pos_tag):
    if pos_tag.startswith('V'):
        return 'v'
    elif pos_tag.startswith('N'):
        return 'n'
    elif pos_tag.startswith('R'):
        return 'r'
    elif pos_tag.startswith('J'):
        return 'a'
    else:
        return "!"

def remove_stop_words(path):
    target_dir = r"C:\Users\Delia\Desktop\AN3\sem1\ML\ML-project\data\proccessed_data"
    target_dir_name = os.path.basename(path)
    target_dir_path = os.path.join(target_dir, target_dir_name)
    os.mkdir(target_dir_path)

    stop_words = set(stopwords.words('english'))
    punctuation = set([".", ",", "<", ">", "?", "/", ":", ";", "'", "\"", "{", "}", "[", "]", "\\", "|", "`", "~", "!", "^", "(", ")", "-", "``", "*"])

    lemmatizater = WordNetLemmatizer()

    #Make part dirs and copy filtered content
    for i in range(1,11):
        subdir_name = f"part{i}"
        subdir_path = os.path.join(target_dir_path, subdir_name)
        os.mkdir(subdir_path)

        initial_subdir_path = os.path.join(path, subdir_name)
        for file_name in os.listdir(initial_subdir_path):
            file_path = os.path.join(initial_subdir_path, file_name)
            print(file_path)
            with open(file_path, "r") as file:
                content = file.read()

                #Tokenize the content
                words = word_tokenize(content)
                new_words = []

                for word in words:
                    if not any(char.isdigit() for char in word):
                        word_lower = word.lower()
                        context = adjust_pos_tag(pos_tag([word_lower])[0][1])
                        lemmatized_word = word_lower

                        if context != "!":
                            lemmatized_word = lemmatizater.lemmatize(word_lower, context)

                        if lemmatized_word not in stop_words and lemmatized_word not in punctuation and lemmatized_word not in new_words:
                            new_words.append(lemmatized_word)

                #Write new words into a new file
                new_file_path = os.path.join(subdir_path, file_name)
                with open(new_file_path, "w") as new_file:
                    for word in new_words:
                        new_file.write(word + "\n")
                print("File created")

def extract_atributes(path):
    #attributes represents a dictionary with string keys represents word and value is represented by a list,
    # where list[0] - number of appearances of current word in non-spam emails and
    # list[1] - number of appearances of current word in spam emails
    attributes = {}
    for root, subdirs, files in os.walk(path):

        #Verify if directory is for training
        if not root.endswith("part10"):
            for file_name in files:
                is_spam = False
                file_path = os.path.join(root, file_name)

                #Verify if the file represents a spam email
                if "spm" in file_name:
                    is_spam = True

                #Extract the words from the file and add them into attributes dictionary
                with open(file_path, "r") as file:
                    content = file.read()

                    # Tokenize the content
                    words = word_tokenize(content)

                    for word in words:
                        if word in attributes:
                            new_list = attributes.get(word)
                            if not is_spam:
                                new_list[0] += 1
                            else:
                                new_list[1] += 1
                            attributes[word] = new_list
                        else:
                            if not is_spam:
                                lst = [1, 0]
                            else:
                                lst = [0, 1]
                            attributes[word] = lst

    #Save dictionary into a file
    with open(r"C:\Users\Delia\Desktop\AN3\sem1\ML\ML-project\data\atributes\dictionary.json", "w") as json_file:
        json.dump(attributes, json_file)
def get_number_of_spam_notspam_instances(path):
    spam_instances = 0
    non_spam_instances = 0
    for root, subdirs, files in os.walk(path):

        # Verify if directory is for training
        if not root.endswith("part10"):
            for file_name in files:
                # Verify if the file represents a spam email
                if "spm" in file_name:
                    spam_instances += 1
                else:
                    non_spam_instances += 1
    return non_spam_instances, spam_instances
def Bayes_Naive(email_path, attributes, number_of_non_spam_instances, number_of_spam_instances):
    total_number_instances = number_of_spam_instances + number_of_non_spam_instances
    probability_non_spam = number_of_non_spam_instances / total_number_instances
    probability_spam = number_of_spam_instances / total_number_instances

    laplace = False

    with open(email_path, "r") as instance:
        content = instance.read()

        # Tokenize the content
        instance_words = word_tokenize(content)

        #Verify if needs Laplace rule
        for attribute in attributes:
            if attribute in instance_words:
                probability_non_spam *= (attributes[attribute][0] / number_of_non_spam_instances)
                probability_spam *= (attributes[attribute][1] / number_of_spam_instances)
            else:
                probability_non_spam *= (1 - (attributes[attribute][0] / number_of_non_spam_instances))
                probability_spam *= (1 - (attributes[attribute][1] / number_of_spam_instances))
            if probability_non_spam == 0 or probability_spam == 0:
                laplace = True
                break

        probability_non_spam = number_of_non_spam_instances / total_number_instances
        probability_spam = number_of_spam_instances / total_number_instances

        for attribute in attributes:
            if attribute in instance_words:
                if laplace:
                    probability_non_spam *= ((attributes[attribute][0] + 1) / (number_of_non_spam_instances + 2))
                    probability_spam *= ((attributes[attribute][1] + 1) / (number_of_spam_instances + 2))
                else:
                    probability_non_spam *= (attributes[attribute][0] / number_of_non_spam_instances)
                    probability_spam *= (attributes[attribute][1] / number_of_spam_instances)
            else:
                if laplace:
                    probability_non_spam *= (1 - ((attributes[attribute][0] + 1) / (number_of_non_spam_instances + 2)))
                    probability_spam *= (1 - ((attributes[attribute][1] + 1) / (number_of_spam_instances + 2)))
                else:
                    probability_non_spam *= (1 - (attributes[attribute][0] / number_of_non_spam_instances))
                    probability_spam *= (1 - (attributes[attribute][1] / number_of_spam_instances))

            if probability_non_spam < 1e-10 and probability_spam < 1e-10:
                probability_non_spam *= 10**6
                probability_spam *= 10**6

    print(f"Prob for non spam:{probability_non_spam}")
    print(f"Prob for spam:{probability_spam}")
    print(laplace)

    if probability_non_spam > probability_spam:

        print("Email is not spam!")
    else:
        print("Email is spam!")




#if __name__ == "main":
# reset_folder(r"C:\Users\Delia\Desktop\AN3\sem1\ML\ML-project\data\proccessed_data")
# remove_stop_words(r"C:\Users\Delia\Desktop\AN3\sem1\ML\ML-project\data\lingspam_public\bare")
# remove_stop_words(r"C:\Users\Delia\Desktop\AN3\sem1\ML\ML-project\data\lingspam_public\lemm")
# remove_stop_words(r"C:\Users\Delia\Desktop\AN3\sem1\ML\ML-project\data\lingspam_public\lemm_stop")
# remove_stop_words(r"C:\Users\Delia\Desktop\AN3\sem1\ML\ML-project\data\lingspam_public\stop")
#extract_atributes(r"C:\Users\Delia\Desktop\AN3\sem1\ML\ML-project\data\proccessed_data")
number_of_non_spam_instances, number_of_spam_instances = get_number_of_spam_notspam_instances(r"C:\Users\Delia\Desktop\AN3\sem1\ML\ML-project\data\proccessed_data")
# print(number_of_non_spam_instances)
# print(number_of_spam_instances)

#Read dictionary from file
with open(r"C:\Users\Delia\Desktop\AN3\sem1\ML\ML-project\data\atributes\dictionary.json", "r") as json_file:
    attributes = json.load(json_file)

Bayes_Naive(r"C:\Users\Delia\Desktop\AN3\sem1\ML\ML-project\data\proccessed_data\lemm_stop\part10\spmsgc99.txt", attributes, number_of_non_spam_instances, number_of_spam_instances)


