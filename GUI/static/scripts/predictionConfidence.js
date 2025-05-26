const labels = ["Fleche", "Lunge", "En Garde", "Step"];
const initialData = [0.5, 0.2, 0.8, 0.1];

const confidenceContext = document.getElementById("predictionChart");

const myChart = new Chart(confidenceContext, {
  type: "bar",
  data: {
    labels: labels,
    datasets: [
      {
        label: "Confidence",
        data: initialData,
        backgroundColor: "rgba(75, 192, 192, 0.5)",
        borderColor: "rgba(75, 192, 192, 1)",
        borderWidth: 1,
        barPercentage: 1.0,
        categoryPercentage: 1.0,
      },
    ],
  },
  options: {
    animation: {
      duration: 800,
    },
    scales: {
      y: {
        beginAtZero: true,
        min: 0,
        title: {
          display: true,
          text: "Confidence (%)",
        },
      },
    },
  },
});

// This function accepts decimal values (0.0 to 1.0)
function updateChart(fleche, lunge, enGarde, step) {
  const inputs = [fleche, lunge, enGarde, step];
  myChart.data.datasets[0].data = inputs;
  myChart.update();
}

// Example usage: bars will reflect actual values (0.5 = 50%)
// updateChart(0.5, 0.2, 0.8, 0.1);

updateChart(10, 20, 40, 40);
