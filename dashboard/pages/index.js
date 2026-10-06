import { useEffect, useState } from "react";
import Head from "next/head";
import patientsData from "../data/patients.json";
import Header from "../components/Header";
import PatientInfoBar from "../components/PatientInfoBar";
import RiskChart from "../components/RiskChart";
import DriversPanel from "../components/DriversPanel";
import VitalsPanel from "../components/VitalsPanel";
import LabsPanel from "../components/LabsPanel";
import NotesTimeline from "../components/NotesTimeline";
import SummaryPanel from "../components/SummaryPanel";
import AISuggestions from "../components/AISuggestions";
import PhysicianNotes from "../components/PhysicianNotes";
import Footer from "../components/Footer";

export default function Home() {
  const { patients } = patientsData;
  const [selectedId, setSelectedId] = useState(patients[0]?.id);
  const [hoverIdx, setHoverIdx] = useState(null);
  const patient = patients.find((p) => p.id === selectedId) || patients[0];

  // reset the hovered hour whenever the patient changes, so a leftover
  // hover index from a longer stay doesn't point past the end of a shorter one
  useEffect(() => setHoverIdx(null), [selectedId]);

  if (!patient) {
    return (
      <div className="page">
        <p>No patient data found. Run export_dashboard_data.py to generate data/patients.json.</p>
      </div>
    );
  }

  const latestIdx = patient.hours.length - 1;
  const currentRisk = patient.risk[latestIdx];

  return (
    <>
      <Head>
        <title>EHR-Matrix AI</title>
      </Head>
      <div className="page">
        <Header patients={patients} selectedId={patient.id} onSelect={setSelectedId} />
        <PatientInfoBar patient={patient} currentRisk={currentRisk} />
        <div style={{ marginTop: 14 }}>
          <RiskChart patient={patient} hoverIdx={hoverIdx} onHover={setHoverIdx} />
        </div>
        <DriversPanel patient={patient} hoverIdx={hoverIdx} />
        <div className="row">
          <VitalsPanel patient={patient} hourIdx={latestIdx} />
          <LabsPanel patient={patient} hourIdx={latestIdx} />
        </div>
        <NotesTimeline patient={patient} />
        <SummaryPanel patient={patient} />
        <div className="row">
          <AISuggestions patient={patient} />
          <PhysicianNotes patientId={patient.id} />
        </div>
        <Footer />
      </div>
    </>
  );
}