import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main():

    df = load_data()

    df_airbus = pd.read_csv('airbus.csv', sep=",")
    df_atos = pd.read_csv('atos.csv', sep=",")
    df_renault = pd.read_csv('renault.csv', sep=",")
    df_sanofi = pd.read_csv('sanofi.csv', sep=",")

    df_johnsonjohnson = pd.read_csv('johnsonjohnson.csv', sep=",")
    df_visa = pd.read_csv('visa.csv', sep=",")
    df_jpmorgan = pd.read_csv('jpmorgan.csv', sep=",")
    df_berkshirehathaway = pd.read_csv('berkshirehathaway.csv', sep=",")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    page = st.sidebar.selectbox("Choose a page", ['Homepage', 'Exploration', 'Prediction'])



    if page == 'Homepage':
        st.title('Sociétés françaises :')
        st.text('Airbus')
        st.text(f"Nombre de tweets trouvées pour Airbus = {len(df_airbus)}\nMoyenne des sentiments = {df_airbus['polarity_spacy'].mean()}\nTotal comments = {df_airbus.comment_num.sum()}\nTotal retweet = {df_airbus.retweet_num.sum()}\nTotal like = {df_airbus.like_num.sum()}")
        st.dataframe(df_airbus)
        sns.lineplot(x='post_date', y='polarity_spacy', data=df_airbus)
        st.pyplot()

        st.text('Atos')
        st.text(f"Nombre de tweets trouvées pour Atos = {len(df_atos)}\nMoyenne des sentiments = {df_atos['polarity_spacy'].mean()}\nTotal comments = {df_atos.comment_num.sum()}\nTotal retweet = {df_atos.retweet_num.sum()}\nTotal like = {df_atos.like_num.sum()}")
        st.dataframe(df_atos)
        sns.lineplot(x='post_date', y='polarity_spacy', data=df_atos)
        st.pyplot()

        st.text('Renault')
        st.text(f"Nombre de tweets trouvées pour Renault = {len(df_renault)}\nMoyenne des sentiments = {df_renault['polarity_spacy'].mean()}\nTotal comments = {df_renault.comment_num.sum()}\nTotal retweet = {df_renault.retweet_num.sum()}\nTotal like = {df_renault.like_num.sum()}")
        st.dataframe(df_renault)
        sns.lineplot(x='post_date', y='polarity_spacy', data=df_renault)
        st.pyplot()

        st.text('Sanofi')
        st.text(f"Nombre de tweets trouvées pour Sanofi = {len(df_sanofi)}\nMoyenne des sentiments = {df_sanofi['polarity_spacy'].mean()}\nTotal comments = {df_sanofi.comment_num.sum()}\nTotal retweet = {df_sanofi.retweet_num.sum()}\nTotal like = {df_sanofi.like_num.sum()}")
        st.dataframe(df_sanofi)
        sns.lineplot(x='post_date', y='polarity_spacy', data=df_sanofi)
        st.pyplot()



        st.title('Sociétés américaines :')
        st.text('Johnson&Johnson')
        st.text(f"Nombre de tweets trouvées pour Johnson = {len(df_johnsonjohnson)}\nMoyenne des sentiments = {df_johnsonjohnson['polarity_spacy'].mean()}\nTotal comments = {df_johnsonjohnson.comment_num.sum()}\nTotal retweet = {df_johnsonjohnson.retweet_num.sum()}\nTotal like = {df_johnsonjohnson.like_num.sum()}")
        st.dataframe(df_johnsonjohnson)
        sns.lineplot(x='post_date', y='polarity_spacy', data=df_johnsonjohnson)
        st.pyplot()

        st.text('Visa')
        st.text(f"Nombre de tweets trouvées pour Visa = {len(df_visa)}\nMoyenne des sentiments = {df_visa['polarity_spacy'].mean()}\nTotal comments = {df_visa.comment_num.sum()}\nTotal retweet = {df_visa.retweet_num.sum()}\nTotal like = {df_visa.like_num.sum()}")
        st.dataframe(df_visa)
        sns.lineplot(x='post_date', y='polarity_spacy', data=df_visa)
        st.pyplot()

        st.text('JPMorgan')
        st.text(f"Nombre de tweets trouvées pour JPMorgan = {len(df_jpmorgan)}\nMoyenne des sentiments = {df_jpmorgan['polarity_spacy'].mean()}\nTotal comments = {df_jpmorgan.comment_num.sum()}\nTotal retweet = {df_jpmorgan.retweet_num.sum()}\nTotal like = {df_jpmorgan.like_num.sum()}")
        st.dataframe(df_jpmorgan)
        sns.lineplot(x='post_date', y='polarity_spacy', data=df_jpmorgan)
        st.pyplot()

        st.text('BerkshireHathaway')
        st.text(f"Nombre de tweets trouvées pour BerkshireHathaway = {len(df_berkshirehathaway)}\nMoyenne des sentiments = {df_berkshirehathaway['polarity_spacy'].mean()}\nTotal comments = {df_berkshirehathaway.comment_num.sum()}\nTotal retweet = {df_berkshirehathaway.retweet_num.sum()}\nTotal like = {df_berkshirehathaway.like_num.sum()}")
        st.dataframe(df_berkshirehathaway)
        sns.lineplot(x='post_date', y='polarity_spacy', data=df_berkshirehathaway)
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
    else:
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