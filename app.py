import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# uvicorn api:app --reload    #streamlit run app.py
#procfil : web: uvicorn api:app --host=0.0.0.0 --port=${PORT:-5000}        web: sh setup.sh && streamlit run api_sl.py    

def main():

    df = load_data()
    df_main = pd.read_csv('tweets_100.csv', sep=",")

    df_airbus = pd.read_csv('airbus.csv', sep=",")
    df_atos = pd.read_csv('atos.csv', sep=",")
    df_renault = pd.read_csv('renault.csv', sep=",")
    df_sanofi = pd.read_csv('sanofi.csv', sep=",")

    df_johnsonjohnson = pd.read_csv('johnsonjohnson.csv', sep=",")
    df_visa = pd.read_csv('visa.csv', sep=",")
    df_jpmorgan = pd.read_csv('jpmorgan.csv', sep=",")
    df_berkshirehathaway = pd.read_csv('berkshirehathaway.csv', sep=",")

    df_total_fr = pd.concat([df_airbus, df_atos, df_renault, df_sanofi], ignore_index=True,)
    df_total_us = pd.concat([df_johnsonjohnson, df_visa, df_jpmorgan, df_berkshirehathaway], ignore_index=True,)
    
    df_total_fr.to_csv('df_total_fr.csv')
    df_total_us.to_csv('df_total_us.csv')

    st.set_option('deprecation.showPyplotGlobalUse', False)
    page = st.sidebar.selectbox("Selectionne une page", ['Données récupérées', 'Sociétés françaises', 'Sociétés américaines','Comparaisons FR vs US'])

    if page == 'Données récupérées':
        st.title('Données récupérées')
        st.dataframe(df_main[0:100])

    if page == 'Sociétés françaises':
        st.title('Sociétés françaises données de 2019.06 à 2019.12 :')
        st.title('Airbus')
        st.text(f"Nombre de tweets trouvées pour Airbus = {len(df_airbus)}\nMoyenne des sentiments = {df_airbus['polarity_spacy'].mean()}\nTotal comments = {df_airbus.comment_num.sum()}\nTotal retweet = {df_airbus.retweet_num.sum()}\nTotal like = {df_airbus.like_num.sum()}")
        if st.checkbox('Montrer le dataframe Airbus'):
            st.dataframe(df_airbus)
        if st.checkbox('Montrer le graphique polarity_spacy Airbus'):
            sns.lineplot(x='post_date', y='polarity_spacy', data=df_airbus).set(xticklabels=[])
            st.pyplot()

        st.title('Atos')
        st.text(f"Nombre de tweets trouvées pour Atos = {len(df_atos)}\nMoyenne des sentiments = {df_atos['polarity_spacy'].mean()}\nTotal comments = {df_atos.comment_num.sum()}\nTotal retweet = {df_atos.retweet_num.sum()}\nTotal like = {df_atos.like_num.sum()}")
        if st.checkbox('Montrer le dataframe Atos'):
            st.dataframe(df_atos)
        if st.checkbox('Montrer le graphique polarity_spacy Atos'):
            sns.lineplot(x='post_date', y='polarity_spacy', data=df_atos).set(xticklabels=[])
            st.pyplot()

        st.title('Renault')
        st.text(f"Nombre de tweets trouvées pour Renault = {len(df_renault)}\nMoyenne des sentiments = {df_renault['polarity_spacy'].mean()}\nTotal comments = {df_renault.comment_num.sum()}\nTotal retweet = {df_renault.retweet_num.sum()}\nTotal like = {df_renault.like_num.sum()}")
        if st.checkbox('Montrer le dataframe Renault'):
            st.dataframe(df_renault)
        if st.checkbox('Montrer le graphique polarity_spacy Renault'):
            sns.lineplot(x='post_date', y='polarity_spacy', data=df_renault).set(xticklabels=[])
            st.pyplot()

        st.title('Sanofi')
        st.text(f"Nombre de tweets trouvées pour Sanofi = {len(df_sanofi)}\nMoyenne des sentiments = {df_sanofi['polarity_spacy'].mean()}\nTotal comments = {df_sanofi.comment_num.sum()}\nTotal retweet = {df_sanofi.retweet_num.sum()}\nTotal like = {df_sanofi.like_num.sum()}")
        if st.checkbox('Montrer le dataframe Sanofi'):
            st.dataframe(df_sanofi)
        if st.checkbox('Montrer le graphique polarity_spacy Sanofi'):
            sns.lineplot(x='post_date', y='polarity_spacy', data=df_sanofi).set(xticklabels=[])
            st.pyplot()


    if page == 'Sociétés américaines':
        st.title('Sociétés américaines données de 2019.06 à 2019.12 :')
        st.title('Johnson&Johnson')
        st.text(f"Nombre de tweets trouvées pour Johnson = {len(df_johnsonjohnson)}\nMoyenne des sentiments = {df_johnsonjohnson['polarity_spacy'].mean()}\nTotal comments = {df_johnsonjohnson.comment_num.sum()}\nTotal retweet = {df_johnsonjohnson.retweet_num.sum()}\nTotal like = {df_johnsonjohnson.like_num.sum()}")
        if st.checkbox('Montrer le dataframe Johnson&Johnson'):
            st.dataframe(df_johnsonjohnson)
        if st.checkbox('Montrer le graphique polarity_spacy Johnson&Johnson'):
            sns.lineplot(x='post_date', y='polarity_spacy', data=df_johnsonjohnson).set(xticklabels=[])
            st.pyplot()

        st.title('Visa')
        st.text(f"Nombre de tweets trouvées pour Visa = {len(df_visa)}\nMoyenne des sentiments = {df_visa['polarity_spacy'].mean()}\nTotal comments = {df_visa.comment_num.sum()}\nTotal retweet = {df_visa.retweet_num.sum()}\nTotal like = {df_visa.like_num.sum()}")
        if st.checkbox('Montrer le dataframe Visa'):
            st.dataframe(df_visa)
        if st.checkbox('Montrer le graphique polarity_spacy Visa'):
            sns.lineplot(x='post_date', y='polarity_spacy', data=df_visa).set(xticklabels=[])
            st.pyplot()

        st.title('JPMorgan')
        st.text(f"Nombre de tweets trouvées pour JPMorgan = {len(df_jpmorgan)}\nMoyenne des sentiments = {df_jpmorgan['polarity_spacy'].mean()}\nTotal comments = {df_jpmorgan.comment_num.sum()}\nTotal retweet = {df_jpmorgan.retweet_num.sum()}\nTotal like = {df_jpmorgan.like_num.sum()}")
        if st.checkbox('Montrer le dataframe JPMorgan'):
            st.dataframe(df_jpmorgan)
        if st.checkbox('Montrer le graphique polarity_spacy JPMorgan'):
            sns.lineplot(x='post_date', y='polarity_spacy', data=df_jpmorgan).set(xticklabels=[])
            st.pyplot()

        st.title('BerkshireHathaway')
        st.text(f"Nombre de tweets trouvées pour BerkshireHathaway = {len(df_berkshirehathaway)}\nMoyenne des sentiments = {df_berkshirehathaway['polarity_spacy'].mean()}\nTotal comments = {df_berkshirehathaway.comment_num.sum()}\nTotal retweet = {df_berkshirehathaway.retweet_num.sum()}\nTotal like = {df_berkshirehathaway.like_num.sum()}")
        if st.checkbox('Montrer le dataframe BerkshireHathaway'):
            st.dataframe(df_berkshirehathaway)
        if st.checkbox('Montrer le graphique polarity_spacy BerkshireHathaway'):
            sns.lineplot(x='post_date', y='polarity_spacy', data=df_berkshirehathaway).set(xticklabels=[])
            st.pyplot()


    if page == 'Comparaisons FR vs US':
        st.title('Comparaisons sociétés françaises et américaines données de 2019.06 à 2019.12 :')
        
        st.title('Total Sociétés Françaises')
        st.text(f"Nombre total de tweets trouvées pour les sociétés françaises = {len(df_total_fr)} \nMoyenne des sentiments = {df_total_fr['polarity_spacy'].mean()} \nTotal comments = {df_total_fr.comment_num.sum()} \nTotal retweet = {df_total_fr.retweet_num.sum()} \nTotal like = {df_total_fr.like_num.sum()}")
        if st.checkbox('Montrer le dataframe de Sociétés Françaises'):
            st.dataframe(df_total_fr)
        if st.checkbox('Montrer le graphique polarity_spacy Sociétés Françaises'):
            sns.lineplot(x='post_date', y='polarity_spacy', data=df_total_fr).set(xticklabels=[])
            st.pyplot()
        if st.checkbox('graphique polarity_spacy Fr'): 
            df_total_fr.hist()
            plt.show()
            st.pyplot()
            
        st.title('Total Sociétés Américaines')
        st.text(f"Nombre total de tweets trouvées pour les sociétés Américaines = {len(df_total_us)} \nMoyenne des sentiments = {df_total_us['polarity_spacy'].mean()} \nTotal comments = {df_total_us.comment_num.sum()} \nTotal retweet = {df_total_us.retweet_num.sum()} \nTotal like = {df_total_us.like_num.sum()}")
        if st.checkbox('Montrer le dataframe de Sociétés Américaines'):
            st.dataframe(df_total_us)
        if st.checkbox('Montrer le graphique polarity_spacy Sociétés Américaines'):
            sns.lineplot(x='post_date', y='polarity_spacy', data=df_total_us).set(xticklabels=[])
            st.pyplot()
        if st.checkbox('graphique polarity_spacy US'): 
            df_total_us.hist()
            plt.show()
            st.pyplot()



    elif page == 'Exploration':
        st.title('Explore the Wine Data-set')
        if st.checkbox('Show column descriptions'):
            st.dataframe(df.describe())
        
        st.markdown('### Analysing column relations')
        st.text('Correlations:')
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(df.corr(), annot=True, ax=ax)
        st.pyplot(fig)
        st.text('Effect of the different classes')
        fig = sns.pairplot(df, vars=['magnesium', 'flavanoids', 'nonflavanoid_phenols', 'proline'], hue='alcohol')
        st.pyplot(fig)
    elif page == 'page3':
        st.title('Modelling')
        model, accuracy = train_model(df)
        st.write('Accuracy: ' + str(accuracy))
        st.markdown('### Make prediction')
        st.dataframe(df)
        row_number = st.number_input('Select row', min_value=0, max_value=len(df)-1, value=0)
        st.markdown('#### Predicted')
        st.text(model.predict(df.drop(['alcohol'], axis=1).loc[row_number].values.reshape(1, -1))[0])


@st.cache(allow_output_mutation=True)
def train_model(df):
    X = np.array(df.drop(['alcohol'], axis=1))
    y= np.array(df['alcohol'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model, model.score(X_test, y_test)

@st.cache
def load_data():
    return pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', names=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols','flavanoids', 'nonflavanoid_phenols' ,'proanthocyanins', 'color_intensity', 'hue', 'OD280/OD315_of_diluted_wines', 'proline'], delimiter=",", index_col=False)


if __name__ == '__main__':
    main()