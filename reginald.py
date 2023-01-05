
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.pipelines import pipeline
import random

st.set_page_config(page_title="Reginald von Doom", page_icon = "🤖", initial_sidebar_state = 'auto')

data = [
    """The United States Declaration of Independence was the first Etext
released by Project Gutenberg, early in 1971.  The title was stored
in an emailed instruction set which required a tape or diskpack be
hand mounted for retrieval.  The diskpack was the size of a large
cake in a cake carrier, cost $1500, and contained 5 megabytes, of
which this file took 1-2%.  Two tape backups were kept plus one on
paper tape.  The 10,000 files we hope to have online by the end of
2001 should take about 1-2% of a comparably priced drive in 2001.""",
"""No person shall be held to answer for a capital, or otherwise infamous crime,
unless on a presentment or indictment of a Grand Jury, except in cases arising
in the land or naval forces, or in the Militia, when in actual service
in time of War or public danger; nor shall any person be subject for
the same offense to be twice put in jeopardy of life or limb;
nor shall be compelled in any criminal case to be a witness against himself,
nor be deprived of life, liberty, or property, without due process of law;
nor shall private property be taken for public use without just compensation.""",
"""The law regarding corporations prescribes that a corporation can be incorporated in the state of Montana to serve any lawful purpose.  In the state of Montana, a corporation has all the powers of a natural person for carrying out its business activities.  The corporation can sue and be sued in its corporate name.  It has perpetual succession.  The corporation can buy, sell or otherwise acquire an interest in a real or personal property.  It can conduct business, carry on operations, and have offices and exercise the powers in a state, territory or district in possession of the U.S., or in a foreign country.  It can appoint officers and agents of the corporation for various duties and fix their compensation.
The name of a corporation must contain the word “corporation” or its abbreviation “corp.”  The name of a corporation should not be deceptively similar to the name of another corporation incorporated in the same state.  It should not be deceptively identical to the fictitious name adopted by a foreign corporation having business transactions in the state.
The corporation is formed by one or more natural persons by executing and filing articles of incorporation to the secretary of state of filing""",
"""To the Congress of the United States:
Where we choose to invest speaks to what we value as a Nation.
This year’s Budget, the first of my Presidency, is a statement of values that define our Nation at
its best. It is a Budget for what our economy can be, who our economy can serve, and how we can
build it back better by putting the needs, goals, ingenuity, and strength of the American people front
and center.
The Budget is built around a fundamental understanding of how our economy works and why, for
too long and for too many, it has not. It is a Budget that reflects the fact that trickle-down economics
has never worked, and that the best way to grow our economy is not from the top down, but from
the bottom up and the middle out. Our prosperity comes from the people who get up every day, work
hard, raise their family, pay their taxes, serve their Nation, and volunteer in their communities. If
we make that understanding our foundation, everything we build upon it will be strong.""",
"""National Commercial Bank (NCB), Saudi Arabia’s largest lender by assets, agreed to buy rival Samba Financial Group 
for $15 billion in the biggest banking takeover this year.NCB will pay 28.45 riyals ($7.58) for each Samba share, 
according to a statement on Sunday, valuing it at about 55.7 billion riyals. NCB will offer 0.739 new shares for each 
Samba share, at the lower end of the 0.736-0.787 ratio the banks set when they signed an initial framework agreement 
in June.The offer is a 3.5% premium to Samba’s Oct. 8 closing price of 27.50 riyals and about 24% higher than the 
level the shares traded at before the talks were made public. Bloomberg News first reported the merger discussions.The new 
bank will have total assets of more than $220 billion, creating the Gulf region’s third-largest lender. The entity’s 
$46 billion market capitalization nearly matches that of Qatar National Bank QPSC, which is still the Middle East’s 
biggest lender with about $268 billion of assets."""
]

docs = {
    "The United States Declaration of Independence": data[0],
    "The Fifth Ammendment": data[1],
    "Montana Corporation Law": data[2],
    "President Biden's Remarks on Budget 2022": data[3],
    "NCB and Samba Merger": data[4]
}

st.cache(show_spinner=True)
def load_model():
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    nlp_pipeline = pipeline("summarization", model=model, tokenizer=tokenizer)
    return nlp_pipeline

nlp_pipeline = load_model()

st.header("🤖 Reginald von Doom 🤖")
st.text("Your personal legal and financial assistant")

st.subheader("👇🏽 Your Legal/Financial Document Goes Here 👇🏽")

selected_document = st.selectbox(
    "Select a random document from the dropdown", 
    docs.keys()
)

text = st.text_area(label="", value=docs[selected_document], height=250, max_chars=None, key=None)
button = st.button("⚙️ Generate Summary ⚙️")

if not len(text) == 0:
    preprocessed_text = text.strip().replace("\n", "")
    t5_prepared_text = "summarize: " + preprocessed_text
    x_dict = nlp_pipeline(t5_prepared_text)
    st.subheader("💎 Summary 💎")
    if button:
        st.write(x_dict[0]["summary_text"])