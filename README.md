#Backend
cd server/backend
pip install -r requirements_minimal.txt
.env   # create under backend:
 GROQ_API_KEY=your_key_here //generate one and put
 SECRET_KEY=any_random_string
 DATABASE_URL=sqlite:///./cogniwave.db
python main.py


#Frontend
cd frontend/client
npm install    # once
npm run dev
