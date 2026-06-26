from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    topic = """
A closure is the combination of a function bundled together (enclosed) with references to its surrounding state (the lexical environment). In other words, a closure gives a function access to its outer scope. In JavaScript, closures are created every time a function is created, at function creation time.
    """

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = PromptTemplate(
        input_variables=["topic"],
        template=(
            "Summarize the following in this exact format:\n"
            "Summary: <one sentence>\n"
            "Point 1: <key point>\n"
            "Point 2: <key point>\n\n"
            "Text: {topic}"
        ),
    )

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"topic": topic})
    print(result)


if __name__ == "__main__":
    main()
