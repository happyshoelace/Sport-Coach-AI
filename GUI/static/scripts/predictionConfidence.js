const labels = ["En Garde", "Fleche", "Lunge", "Step"]; // Match backend order!
const allPredictions = window.all_predictions || [];

if (allPredictions.length === 0) {
  // fallback
  renderChart([0, 0, 0, 0]);
} else {
  // Sum up the predictions for each class
  const classTotals = [0, 0, 0, 0];
  allPredictions.forEach((arr) => {
    arr.forEach((val, idx) => {
      classTotals[idx] += val;
    });
  });

  // Normalize to percentage
  const totalFrames = allPredictions.length;
  const classPercentages = classTotals.map((count) => (count / totalFrames) * 100);

  renderChart(classPercentages);
}

function renderChart(data) {
  const confidenceContext = document.getElementById("predictionChart");
  new Chart(confidenceContext, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Prediction Confidence (%)",
          data: data,
          backgroundColor: [
            "rgba(255, 99, 132, 0.5)",
            "rgba(54, 162, 235, 0.5)",
            "rgba(255, 206, 86, 0.5)",
            "rgba(75, 192, 192, 0.5)",
          ],
        },
      ],
    },
    options: {
      animation: { duration: 800 },
      scales: {
        y: {
          beginAtZero: true,
          min: 0,
          max: 100,
          title: { display: true, text: "Confidence (%)" },
        },
      },
    },
  });
}
