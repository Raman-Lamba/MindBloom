import React, { useState, useEffect, useRef } from 'react';

function App() {
  const [query, setQuery] = useState('');
  const [conversation, setConversation] = useState([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [conversation]);

  const formatAnswer = (text) => {
    // First handle the markdown headings
    const processedText = text
      .split('\n')
      .map((line, lineIndex) => {
        // Handle headings with ### (h3)
        if (line.startsWith('### ')) {
          return (
            <h3 
              key={`heading-${lineIndex}`} 
              style={{ 
                color: '#68d5f8', 
                fontSize: '1.3rem', 
                fontWeight: '600',
                marginTop: '1.5rem',
                marginBottom: '0.75rem'
              }}
            >
              {line.substring(4)}
            </h3>
          );
        }
        
        // Handle bullet points
        if (line.trim().startsWith('- ') || line.trim().match(/^\d+\.\s/)) {
          return (
            <div 
              key={`bullet-${lineIndex}`} 
              style={{ 
                marginLeft: '1rem',
                marginBottom: '0.5rem' 
              }}
            >
              {line}
            </div>
          );
        }
        
        // Regular text with line breaks
        return (
          <div key={`line-${lineIndex}`} style={{ marginBottom: '0.5rem' }}>
            {line}
          </div>
        );
      });

    // Then handle the bold text with **
    const renderContent = (content) => {
      if (typeof content !== 'string') return content;
      
      return content.split('**').map((part, index) => {
        if (index % 2 === 1) {
          return (
            <strong key={`bold-${index}`} style={{ color: '#68d5f8' }}>
              {part}
            </strong>
          );
        }
        return part;
      });
    };

    return processedText.map(item => {
      if (React.isValidElement(item)) {
        return React.cloneElement(
          item, 
          {...item.props}, 
          typeof item.props.children === 'string' 
            ? renderContent(item.props.children) 
            : item.props.children
        );
      }
      return item;
    });
  };

  const sendQuery = async () => {
    if (!query.trim()) return;
    
    const currentQuery = query;
    setQuery('');
    setLoading(true);
    
    try {
      const response = await fetch('http://localhost:5000/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: currentQuery }),
      });
      const data = await response.json();
      
      setConversation(prev => [
        ...prev,
        { question: currentQuery, answer: data.answer }
      ]);
    } catch (err) {
      setConversation(prev => [
        ...prev,
        { question: currentQuery, answer: 'Something went wrong. Please try again.' }
      ]);
    }
    setLoading(false);
  };

  return (
    <div style={{ 
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      backgroundColor: '#121212',
      fontFamily: "'Segoe UI', Arial, sans-serif",
      position: 'relative',
      color: '#e0e0e0'
    }}>
      <header style={{ 
        padding: '1.5rem',
        backgroundColor: '#1e1e1e',
        boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
        fontSize: '1.5rem',
        fontWeight: '600',
        color: '#68d5f8',
        textAlign: 'center'
      }}>
        MindBloom
      </header>

      {/* Conversation Container */}
      <div style={{ 
        flex: 1,
        overflowY: 'auto',
        padding: '2rem',
        paddingBottom: '120px'
      }}>
        <div style={{ maxWidth: '800px', margin: '0 auto' }}>
          {conversation.map((item, index) => (
            <div key={index} style={{ marginBottom: '2rem' }}>
              <div style={{ 
                backgroundColor: '#1e1e1e',
                borderRadius: '12px',
                padding: '1.5rem',
                marginBottom: '1rem',
                boxShadow: '0 2px 6px rgba(0,0,0,0.2)'
              }}>
                <div style={{ 
                  color: '#a0a0a0',
                  fontSize: '0.9rem',
                  marginBottom: '0.5rem',
                  fontWeight: '500'
                }}>
                  Your question
                </div>
                <p style={{ 
                  margin: 0,
                  color: '#e0e0e0',
                  fontSize: '1.1rem',
                  lineHeight: '1.6'
                }}>
                  {item.question}
                </p>
              </div>
              <div style={{ 
                backgroundColor: '#252525',
                borderRadius: '12px',
                padding: '1.5rem',
                boxShadow: '0 2px 6px rgba(0,0,0,0.2)'
              }}>
                <div style={{ 
                  color: '#68d5f8',
                  fontSize: '0.9rem',
                  marginBottom: '0.5rem',
                  fontWeight: '500'
                }}>
                  Response
                </div>
                <div style={{ 
                  color: '#c0c0c0',
                  fontSize: '1.1rem',
                  lineHeight: '1.6'
                }}>
                  {formatAnswer(item.answer)}
                </div>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Container */}
      <div style={{ 
        position: 'fixed',
        bottom: 0,
        left: 0,
        right: 0,
        padding: '1.5rem',
        backgroundColor: '#1e1e1e',
        boxShadow: '0 -2px 12px rgba(0,0,0,0.2)'
      }}>
        <div style={{ 
          maxWidth: '800px',
          margin: '0 auto',
          position: 'relative'
        }}>
          <textarea
            rows="3"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask your question..."
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendQuery();
              }
            }}
            style={{ 
              width: '100%',
              padding: '1.2rem',
              fontSize: '1rem',
              borderRadius: '10px',
              border: '1px solid #404040',
              backgroundColor: '#2a2a2a',
              color: '#e0e0e0',
              resize: 'vertical',
              minHeight: '60px',
              transition: 'all 0.2s',
              ':focus': {
                outline: 'none',
                borderColor: '#4299e1',
                boxShadow: '0 0 0 3px rgba(66, 153, 225, 0.3)'
              }
            }}
          />
          <button 
            onClick={sendQuery} 
            disabled={loading}
            style={{ 
              position: 'absolute',
              right: '1.2rem',
              bottom: '4.5rem',
              padding: '0.6rem 1.8rem',
              backgroundColor: '#4299e1',
              color: '#ffffff',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              fontSize: '1rem',
              fontWeight: '500',
              transition: 'all 0.2s',
              ':hover': {
                backgroundColor: '#3182ce'
              },
              ':disabled': {
                backgroundColor: '#4a4a4a',
                color: '#8a8a8a',
                cursor: 'not-allowed'
              }
            }}
          >
            {loading ? 'Thinking...' : 'Ask'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;