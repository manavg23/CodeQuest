### CodeQuest

**Overview**  
CodeQuest is an interactive chatbot application that lets users engage with their codebase using Google Gemini. Developed with Streamlit, it provides an intuitive chat interface for querying and exploring code.
!["demo"](./Screenshot%202024-08-29%20at%2012.09.05 AM.png)

**Features**  
- **GitHub Integration**: Users can input their Google Gemini API key and the name of their GitHub repository. The repository is then cloned, segmented, and embedded for interactive use.
- **Chat Interface**: Utilizing Langchain, the application builds a QA retriever, allowing users to ask natural language questions about their codebase.
- **User Interaction**: Engage with the codebase through the chat interface, receiving responses based on the content of the code.

**Usage**  
1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/example/repository.git
   ```
2. **Install Dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment Variables**:  
   - Add your Google Gemini API Key to the `.env` file.
   - Obtain and set your Deeplake API key.
4. **Run the Streamlit App**:  
   ```bash
   streamlit run chatbot.py
   ```
5. **Access the Chat Interface**: Open your browser and navigate to [http://localhost:8501](http://localhost:8501). Enter your Google Gemini key and GitHub repository name to start interacting with your codebase.

**Limitations**  
- **Performance**: Large or intricate codebases may take longer to process.
- **Accuracy**: The quality of responses is influenced by the capabilities of Google Gemini and the code embeddings.

**Future Improvements**  
- **Advanced Integration**: Incorporate additional tools for more comprehensive codebase analysis.
- **Feature Expansion**: Enhance functionalities based on user feedback and new requirements.

**Contributing**  
Contributions are encouraged! Report issues or submit pull requests to help improve the project.

**License**  
This project is licensed under the MIT License.

**Acknowledgements**  
Inspired by the power of Google Gemini and Langchain, this tool aims to enhance codebase interaction. Special thanks to all contributors and maintainers of the libraries and frameworks used in this project.
