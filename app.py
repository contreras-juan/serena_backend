from fastapi import FastAPI, Request, UploadFile, File
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.tools import tool 
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import InMemorySaver
from openai import OpenAI

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)



model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7
)

prompt = """
Tu nombre es Serena. Eres una acompañante emocional. Tu propósito es brindar apoyo emocional, contención, escucha empática y sugerencias prácticas para el bienestar mental, sin hacer diagnósticos clínicos ni emitir juicios.

Usa un lenguaje amable, cálido y cercano. Ayuda a las personas a explorar lo que sienten, valida sus emociones y fomenta la autorreflexión. Puedes ofrecer ejercicios simples de respiración, mindfulness, escritura emocional o preguntas introspectivas suaves.

Si el usuario menciona situaciones de crisis o autolesiones, anima a buscar apoyo profesional o llamar a líneas de ayuda. Nunca emitas diagnósticos ni afirmes que puedes reemplazar a un psicólogo.

Responde siempre en un tono empático, tranquilo y respetuoso. Puedes usar frases como:
- "Estoy aquí para ti"
- "Eso suena difícil, gracias por compartirlo conmigo"
- "¿Quieres que hablemos un poco más de eso?"

Trata de indagar más a profundidad por lo que el usuario está pasando para poder dar apoyo y ayudar a explorar lo que siente.

No te presentes ante el usuario cuándo este no te esté saludando, solo brinda el apoyo emocional que sea pertinente.
Si el usuario te está saludando, sí preséntate. Añade emojis para hacer más cómoda la interacción.

No propongas ningún tipo de ejercicio o estrategia hasta que veas que es completamente necesario para ayudar al usuario

La información base del usuario es la siguiente:
{base_information}
"""

@tool
def breathing_excercise(exercise: str):
    """Vamos a hacer un ejercicio de respiración profunda para controlar la ansiedad"""

    return exercise

@tool
def journal_prompt(exercise: str):
    """"Vamos a escribir en un diario aquellas cosas por las cuáles estamos agradecidos"""

    return exercise

checkpointer = InMemorySaver() 

agent = create_react_agent(
    model,
    tools = [breathing_excercise, journal_prompt],
    debug=False,
    name="Serena"
)

# Modelo para solicitud de usuario
class UserInput(BaseModel):
    message: str

# Inicializar FastAPI
app = FastAPI(title="Serena - Acompañante emocional")

# CORS opcional si usas frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia esto si tienes dominio específico
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ruta principal de interacción con Serena
@app.post("/serena/chat")
async def chat_with_serena(user_input: UserInput):
    
    config = {"configurable": {"thread_id": "thread-1"}}
    try:        
        complete_prompt = prompt.format(base_information=user_input)

        system_dict = {
            "messages": [
                {
                    "role": "user",
                    "content": complete_prompt
                }
            ]
        }
        
        response = agent.invoke(system_dict, config)
        
        response_output = {"user": user_input, "agent": response["messages"][-1].content}
        
        return {"response": response_output}
    
    except Exception as e:
        return {"error": str(e)}

@app.post("/stt/whisper")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        # Guarda temporalmente el archivo
        with open("temp_audio.wav", "wb") as f:
            f.write(contents)
        
        # Abre el archivo para lectura binaria
        with open("temp_audio.wav", "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="es"
            )

        return {"text": transcript.text}
    
    except Exception as e:
        return {"error": str(e)}