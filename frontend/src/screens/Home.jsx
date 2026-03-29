import { useState, useEffect } from "react";
import { supabase } from "../supabase";

function Home({ setScreen, user, onLogout }) {
  const [plants, setPlants] = useState([]);
  const [reminders, setReminders] = useState([]);

  useEffect(() => {
    fetchPlants();
    fetchReminders();
  }, []);

  const fetchPlants = async () => {
    const { data, error } = await supabase
      .from("plants")
      .select("*")
      .eq("user_id", user.id);
    if (!error) setPlants(data);
  };

  const fetchReminders = async () => {
    const { data, error } = await supabase
      .from("reminders")
      .select("*")
      .eq("user_id", user.id);
    if (!error) setReminders(data);
  };

  return (
    <div className="screen">
      <div className="home-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1>Hello Plant Parent 😊🪴</h1>
            <p>How are your plants today?</p>
          </div>
          <button onClick={onLogout} style={{
            background: 'rgba(255,255,255,0.2)',
            border: 'none',
            borderRadius: '8px',
            padding: '6px 12px',
            color: 'white',
            fontSize: '12px',
            cursor: 'pointer',
            fontFamily: 'DM Sans, sans-serif'
          }}>
            Logout
          </button>
        </div>
      </div>

      <div className="home-body">
        <p className="section-title">What do you need?</p>
        <div className="feature-grid">
          <div className="feat-card green" onClick={() => setScreen("chat")}>
            <div className="feat-icon">💬</div>
            <h3>Ask PlantDoc</h3>
            <p>Chat with your AI plant doctor</p>
          </div>
          <div className="feat-card amber" onClick={() => setScreen("identify")}>
            <div className="feat-icon">🔍</div>
            <h3>Identify Plant</h3>
            <p>Learn about any plant</p>
          </div>
          <div className="feat-card blue" onClick={() => setScreen("reminders")}>
            <div className="feat-icon">⏰</div>
            <h3>Reminders</h3>
            <p>Care schedule</p>
          </div>
          <div className="feat-card pink" onClick={() => setScreen("diagnose")}>
            <div className="feat-icon">🚑</div>
            <h3>Disease Check</h3>
            <p>Symptom checker + ML diagnosis</p>
          </div>
        </div>

        <p className="section-title">My Plants</p>
        <div className="plants-row">
          {plants.length === 0 ? (
            <div style={{ width: '100%', textAlign: 'center', padding: '20px', color: '#888', fontSize: '13px' }}>
              No plants added yet 🌱
            </div>
          ) : (
            plants.map((plant) => (
              <div className="plant-card" key={plant.id}>
                <div className="plant-emoji">🪴</div>
                <div className="plant-name">{plant.name}</div>
                <div className="plant-status ok">{plant.species}</div>
              </div>
            ))
          )}
        </div>

        <p className="section-title">Today's Reminders</p>
        <div className="reminder-list">
          {reminders.length === 0 ? (
            <div style={{ width: '100%', textAlign: 'center', padding: '20px', color: '#888', fontSize: '13px' }}>
              No reminders yet ⏰
            </div>
          ) : (
            reminders.map((r) => (
              <div className="reminder-item" key={r.id}>
                <span>{r.care_type?.split(' ')[0]}</span>
                <div>
                  <p>{r.care_type?.split(' ')[1]} {r.plant_name}</p>
                  <span>{r.frequency}</span>
                </div>
                <span className="badge today">Due</span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

export default Home;