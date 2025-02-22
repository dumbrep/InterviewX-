from fastapi import FastAPI,WebSocket,WebSocketDisconnect,File,UploadFile
import openai
import os
from dotenv import load_dotenv
import pyttsx3
from fastapi.middleware.cors import CORSMiddleware
import cv2
import dlib
from scipy.spatial import distance
from deepface import DeepFace
import PyPDF2
import io
import asyncio
from pydantic import BaseModel
import base64



load_dotenv()

openai.api_key= os.getenv("OPENAI_KEY")
app = FastAPI()

origins = [
    "http://localhost", 
    "http://localhost:3000",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)



class ResumeData(BaseModel):
    resume_dt : str
    job_description : str
    jobType :str

jobDescription = ""
resume = ""
job_type = ""


@app.post("/resume")
async def get_resume(resumedt : ResumeData): 
   
    global resume      
    global jobDescription
    global experience_years
    resume = resumedt.resume_dt
    jobDescription = resumedt.job_description
    experience_years = resumedt.jobType

    print(resume)

interation = []

def generate_question(role):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # Ensure you're using the correct model
        messages=[
            {"role": "system", "content": f"""You are an interviewer for the role of {role}{job_type}.Generate the question to ask to the candidate. 
                The previous questions and candidate responses are : {interation}.
                - You have to generate only one question to aks.
                - Analyse previous interaction and generate appropriate question.
                - Maintaion the flow of question answering.
                - *Your first question should be "Tell me about yourself."*
                - If you are going to ask first question (i.e. previous responses are null), it should be "Tell me about yourself"
                - Do not stick with the same flow but ask questions which covers entire aspects of interview from technical questions to H.R. level questions.
                - Below i have give the resume of the candidate, analyze it and generate the proper set of questions
                - Add technical and Non-Technical Questions also
                - Add technical questions also for perticular role
            Note : Generate only a question. Does not include any introduction.
                **Candidate resume ** :{resume} 
                **Job descriptions** : {jobDescription}

            Note : Keep questions alligned with uploaded resume and job descriptions.
             """},
            
        ]
    )
    return response.choices[0].message['content'].strip()


def analyze_response(question, answer):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # Ensure you're using the correct modelsss
        messages=[
            {"role": "system", "content": "You are an expert interview response analyzer."},
            {"role" : "user","content" : f"""You have given the question and it's corresponding response by candidate
                Question : {question}
                Response : {answer}
                Generate best feedback to be given to the candidate
                - Ana
                - If candidate ask you to to answer the previuetion you have aske, give it's ideal response.
                - Give ideal response only when explicitally asked by the candidate.
                - Generate feedback in paragraph format
                - Generate short but swwet feedback 
                - Do not add any introduction
                - Go through Candidate details and job descriptions and alighn your response to them

                **Candidate resume ** :{resume} 
                **Job descriptions** : {jobDescription}

                
            
            **If you think, give ideal response of the question also.Do not say that 'I can provide you ideal response or similar' . If you think, directly state**
            --give ideal response in same paragraph to maintantain the overrall flow of interactions.
                """}
        ]
    )
    return response.choices[0].message['content'].strip()  # Access the correct field


@app.websocket("/video")
async def face_analysis(websocket2: WebSocket):    
    await websocket2.accept()
    
    
  #  await websocket2.send_text("I am connected to you")
    # Load the face detector and facial landmarks predictor from dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Function to calculate the Eye Aspect Ratio (EAR) to check eye closure
    def eye_aspect_ratio(eye):
        # Calculate the Euclidean distances between the two sets of vertical eye landmarks
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        # Calculate the Euclidean distance between the horizontal eye landmarks
        C = distance.euclidean(eye[0], eye[3])
        # Return the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear

    # Set up the video capture (webcam)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        analysis = await asyncio.to_thread(DeepFace.analyze, frame_rgb, actions=["emotion"], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
   
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = detector(gray)

        for face in faces:
            # Get the facial landmarks for the face
            landmarks = predictor(gray, face)
            
            # Get the coordinates for the left and right eye
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

            # Calculate the eye aspect ratio for both eyes
            left_eye_ear = eye_aspect_ratio(left_eye)
            right_eye_ear = eye_aspect_ratio(right_eye)

            # Calculate the average EAR
            ear = (left_eye_ear + right_eye_ear) / 2.0

            # If the EAR is below a certain threshold, it indicates the eyes are closed
            if ear < 0.25:
              
                cv2.putText(frame, "Eye contact not maintained", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

           

        _, buffer = cv2.imencode('.jpg', frame)
        base64_str = base64.b64encode(buffer).decode('utf-8')

        await websocket2.send_text(base64_str)
        #await asyncio.sleep(0.05)  # Adjust frame rate
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    

"""@app.websocket("/video")"""
async def async_face_recognition(websocket2: WebSocket):
    await websocket2.accept()
    asyncio.create_task(face_analysis(websocket2))

@app.websocket("/interview/{role}")
async def interview(websocket: WebSocket, role: str):
        await websocket.accept()
        try:
        # await websocket.send_text(f"welcome to the interview preparation for {role} role!")
            while True:
                question = generate_question(role)
                await websocket.send_text(question)         

                answer = await websocket.receive_text()                
                interation.append({"question": question, "answer": answer})
                feedback = analyze_response(question, answer)
                await websocket.send_text(feedback)
            
        except WebSocketDisconnect:
            print(f"Client disconnected from the interview for role: {role}")
