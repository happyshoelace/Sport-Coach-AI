const classes = ["En Garde", "Fleche", "Lunge", "Step"]; // Match backend order!
const timelineData = window.all_predictions || [];

const ctx = document.getElementById("timelineChart").getContext("2d");

// Get the predicted class index for each frame
const predictedClassIndices = timelineData.map(arr => arr.indexOf(Math.max(...arr)));

// Prepare data for scatter plot (y as class name)
const scatterData = predictedClassIndices.map((classIdx, frameIdx) => ({
  x: frameIdx + 1,
  y: classes[classIdx] // Use class name for y
}));

const timelineChart = new Chart(ctx, {
  type: "scatter",
  data: {
    datasets: [{
      label: "Predicted Class Over Time",
      data: scatterData,
      backgroundColor: "rgba(54, 162, 235, 0.7)",
      pointRadius: 6,
    }]
  },
  options: {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx) => `Frame ${ctx.parsed.x}: ${ctx.parsed.y}`,
        },
      },
    },
    scales: {
      x: {
        type: "linear",
        title: { display: true, text: "Frame" },
        min: 1,
        max: timelineData.length,
        ticks: { stepSize: 1, precision: 0 }
      },
      y: {
        type: "category",
        title: { display: true, text: "Class" },
        labels: classes
      },
    },
  },
});
