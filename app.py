

from flask import Flask, render_template, request, jsonify, session
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")

# Configure Google Gemini API
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    genai.configure(api_key=api_key)
    logger.info("Gemini API configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    raise

class ChatConfig:
    """Configuration constants for the chat system"""
    MAX_HISTORY_LENGTH = 20  # Limit conversation history
    MAX_QUERY_LENGTH = 1000  # Max characters per query
    SYSTEM_PROMPT = """You are a helpful tourist assistant specialized in northern Pakistan regions: Naran, Kaghan, Hunza, etc.

- Respond clearly and concisely
- Provide info on hotels, tourist spots, routes, petrol pumps, and amenities
- When giving directions, provide step-by-step numbered lists
- Use markdown headings (##) for main topics
- Include Google Maps links when location is requested
- Include images if relevant
- Keep tone professional, friendly, and informative
- Politely refuse unrelated questions
- Remember previous conversation context

CRITICAL: If asked, provide distance from current location to destination"""

class TourismKeywords:
    """Tourism-related keywords for relevance checking"""
    KEYWORDS = {
        # General tourism
        'travel', 'trip', 'tourism', 'visit', 'vacation', 'journey', 'tourist', 'explore', 'backpacking', 'guide',
        'itinerary', 'plan', 'planning', 'attractions', 'things to do', 'destination', 'trip planner',
        
        # Locations (Northern Pakistan) - Added missing locations
        'naran', 'manoor valley', 'kaghan', 'mahandri', 'ghanool valley', 'balakot', 'kanshian valley', 'jared',
        'hunza', 'loharbanda', 'gilgit', 'rajwal valley', 'skardu', 'swat', 'murree', 'fairy meadows', 'deosai',
        'khunjerab', 'shogran', 'neelum', 'azad kashmir', 'attabad', 'lake saif ul malook', 'lulusar lake',
        'pak china border', 'babusar', 'jalkhad', 'siri paye', 'ratti gali', 'shounter', 'baltit fort',
        'hoper valley', 'altit fort', 'passu cones', 'eagle nest', 'katpana desert', 'rakaposhi', 'bagrote',
        'shimshal', 'cold desert', 'khaplu', 'machlu', 'minapin', 'ghizer', 'dir', 'chitral', 'mastuj',
        'bisian', 'booni', 'kalam', 'ushu forest','ansoo lake','Ratti Gali Lake','Upper Kachura Lake','Chitta Katha Lake','Rush Lake',
        
        # Major cities and towns (routes to/from northern areas)
        'mansehra', 'abbottabad', 'islamabad', 'rawalpindi', 'peshawar', 'lahore', 'karachi', 'chitral',
        'besham', 'dasu', 'chilas', 'astore', 'minimarg', 'gupis', 'yasin', 'mastuj', 'drosh',
        
        # Accommodations
        'hotel', 'guesthouse', 'motel', 'lodge', 'inn', 'resort', 'rooms', 'room', 'stay', 'accommodation',
        'night stay', 'camping', 'tent', 'campsite', 'book hotel', 'booking', 'hotels in', 'rest house',
        
        # Food & Dining
        'restaurant', 'food', 'dining', 'cafe', 'eat', 'local food', 'dishes', 'pakistani food', 'menu',
        'breakfast', 'lunch', 'dinner', 'buffet', 'desi food', 'tea', 'chai', 'coffee', 'snack', 'bakery',
        
        # Transport & Navigation
        'bus', 'car', 'rent a car', 'jeep', 'driver', 'transport', 'vehicle', 'ride', 'map', 'directions',
        'how to reach', 'road condition', 'route', 'path', 'navigation', 'distance', 'fuel', 'petrol',
        'distance from', 'distance to', 'distance between', 'how far', 'km', 'kilometers', 'miles',
        
        # Activities
        'hiking', 'trekking', 'sightseeing', 'boating', 'rafting', 'climbing', 'fishing', 'skiing',
        'photography', 'wildlife', 'paragliding', 'mountaineering', 'snowfall', 'glacier',
        
        # Weather & Services
        'weather', 'forecast', 'temperature', 'hospital', 'atm', 'bank', 'wifi', 'emergency', 'help',
        
        # Greetings
        'hello', 'hi', 'hey', 'salam', 'aoa', 'assalamualaikum', 'greetings', 'good morning', 'good evening'
    }

class ChatbotService:
    """Service class for chatbot operations"""
    
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")

        self.irrelevant_markers = {
            "i'm here to help with tourism in northern pakistan only",
            "sorry, i can't help with that",
            "i don't have information on that",
            "irrelevant",
            "i cannot answer",
            "not related to tourism"
        }
    
    def is_question_relevant(self, text: str) -> bool:
        """Check if the question is tourism-related"""
        if not text or len(text.strip()) < 2:
            return False
        
        text_lower = text.lower().strip()
        
        # Check if it's a single word location name that we support
        if text_lower in TourismKeywords.KEYWORDS:
            return True
        
        # Check for any tourism-related keywords in the text
        return any(keyword in text_lower for keyword in TourismKeywords.KEYWORDS)
    
    def format_conversation(self, history: List[Dict]) -> str:
        """Format conversation history for the LLM"""
        conversation_parts = [ChatConfig.SYSTEM_PROMPT, "\n--- Conversation History ---"]
        
        for i, msg in enumerate(history):
            role = "Human" if msg["role"] == "user" else "Assistant"
            timestamp = msg.get("timestamp", "")
            conversation_parts.append(f"{role}: {msg['text']}")
        
        conversation_parts.append("--- End History ---\n")
        return "\n".join(conversation_parts)
    
    def generate_response(self, conversation_text: str) -> str:
        """Generate response using Gemini API"""
        try:
            response = self.model.generate_content(
                conversation_text,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1000,
                    top_p=0.8,
                    top_k=40
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def is_response_irrelevant(self, response: str) -> bool:
        """Check if the response indicates irrelevant content"""
        response_lower = response.lower()
        return any(marker in response_lower for marker in self.irrelevant_markers)

# Initialize chatbot service
chatbot_service = ChatbotService()

def get_chat_history() -> List[Dict]:
    """Get chat history from session"""
    return session.get('history', [])

def add_to_history(role: str, text: str) -> None:
    """Add message to chat history with timestamp"""
    if 'history' not in session:
        session['history'] = []
    
    message = {
        "role": role,
        "text": text,
        "timestamp": datetime.now().isoformat()
    }
    
    session['history'].append(message)
    
    # Limit history length to prevent token overflow
    if len(session['history']) > ChatConfig.MAX_HISTORY_LENGTH:
        session['history'] = session['history'][-ChatConfig.MAX_HISTORY_LENGTH:]
    
    session.modified = True

def validate_input(query: str) -> Optional[str]:
    """Validate user input"""
    if not query or not query.strip():
        return "Please enter a question."
    
    if len(query) > ChatConfig.MAX_QUERY_LENGTH:
        return f"Question too long. Please limit to {ChatConfig.MAX_QUERY_LENGTH} characters."
    
    return None

# Routes
@app.route('/')
def home():
    """Home page"""
    return render_template('home.html')

@app.route('/chatbot')
def chatbot():
    """Chatbot page"""
    return render_template('chatbot.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Handle chat questions"""
    try:
        # Get and validate input
        user_query = request.form.get('question', '').strip()
        
        validation_error = validate_input(user_query)
        if validation_error:
            return jsonify({'error': validation_error}), 400
        
        # Check relevance with better context awareness
        if not chatbot_service.is_question_relevant(user_query):
            # Additional check for follow-up responses in context
            history = get_chat_history()
            if history and len(history) > 0:
                # Check if previous conversation was about distance/directions
                recent_messages = history[-3:] if len(history) >= 3 else history
                context_text = " ".join([msg['text'].lower() for msg in recent_messages])
                
                # If recent context contains distance/direction keywords, allow the query
                distance_context_keywords = ['distance', 'direction', 'route', 'how to reach', 'where', 'location']
                if any(keyword in context_text for keyword in distance_context_keywords):
                    # This might be a follow-up location question
                    pass  # Continue processing
                else:
                    refusal_msg = "Sorry, I can only help with tourist-related questions about the northern areas of Pakistan. Please ask about hotels, destinations, routes, or travel information."
                    return jsonify({'response': refusal_msg})
            else:
                refusal_msg = "Sorry, I can only help with tourist-related questions about the northern areas of Pakistan. Please ask about hotels, destinations, routes, or travel information."
                return jsonify({'response': refusal_msg})
        
        # Add user message to history
        add_to_history("user", user_query)
        
        # Generate conversation context
        history = get_chat_history()
        conversation = chatbot_service.format_conversation(history)
        
        # Debug logging (remove in production)
        logger.info(f"User query: {user_query}")
        logger.info(f"History length: {len(history)}")
        
        # Generate response
        bot_reply = chatbot_service.generate_response(conversation)
        
        # Check if response is irrelevant
        if chatbot_service.is_response_irrelevant(bot_reply):
            polite_refusal = "Sorry, I can only help with tourist-related questions about the northern areas of Pakistan. Please ask about travel destinations, accommodations, or local information."
            add_to_history("assistant", polite_refusal)
            return jsonify({'response': polite_refusal})
        
        # Add assistant response to history
        add_to_history("assistant", bot_reply)
        
        return jsonify({'response': bot_reply})
    
    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}")
        error_msg = "I'm experiencing technical difficulties. Please try again in a moment."
        return jsonify({'error': error_msg}), 500

@app.route('/reset', methods=['POST'])
def reset():
    """Reset chat history"""
    try:
        session.pop('history', None)
        logger.info("Chat history reset")
        return jsonify({'status': 'Chat history reset successfully.'})
    except Exception as e:
        logger.error(f"Error resetting chat: {e}")
        return jsonify({'error': 'Failed to reset chat history.'}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Get current chat history (for debugging)"""
    history = get_chat_history()
    return jsonify({
        'history': history,
        'count': len(history)
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Configuration for different environments
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    port = int(os.getenv('PORT', 5000))
    
    logger.info(f"Starting Flask app on port {port} (debug: {debug_mode})")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)





