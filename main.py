from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
        Elon Reeve Musk[1] (AFI:[ˈiːlɒn ˈɹiːv ˈmʌsk]) (Pretoria, 28 giugno 1971) è un imprenditore e politico sudafricano con cittadinanza canadese naturalizzato statunitense.

        Ricopre i ruoli di fondatore, amministratore delegato e direttore tecnico della compagnia aerospaziale SpaceX,[2] fondatore di The Boring Company[3] e della società di intelligenza artificiale xAI, cofondatore di Neuralink e OpenAI,[4] amministratore delegato e product architect della multinazionale automobilistica Tesla,[5] proprietario e presidente di X (precedentemente Twitter).[6] Ha inoltre proposto un sistema di trasporto superveloce conosciuto come Hyperloop One, posta in liquidazione il 21 dicembre 2023.[7] Tramite SpaceX gestisce Starlink, una costellazione di satelliti che fornisce Internet ad alta velocità e bassa latenza a tutto il pianeta.[8]

        Secondo Forbes, al 7 febbraio 2026, con un patrimonio stimato di 849,3 miliardi di dollari, risulta essere la persona più ricca del mondo.[9][10][11]

        Dal 20 gennaio al 29 maggio 2025 è stato a capo del Dipartimento dell'Efficienza Governativa statunitense.
    """
    summary_tempate = f"""
    given the information {information} about a person I want you to create:
    1. A short summary
    2, two interesting facts about them 
    """

    summary_prompt_template = PromptTemplate(
        template=summary_tempate, input_variables=["information"]
    )

    llm = ChatOpenAI(model="gpt-5", temperature=0)
    # llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_prompt_template | llm
    
    response = chain.invoke({"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
