import React from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar } from "react-chartjs-2";
import { fontSize } from "@mui/system";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

export const options = {
  responsive: true,
  plugins: {
    legend: {
      position: "top",
    },
    title: {
      display: true,
      text: "Holdings",
      color: 'rgba(249, 246, 242, 1)'
    },
  },
  scales: {
    x: {
      grid: {
        color: 'rgba(255, 255, 255, 0.3)',
      },
      ticks: {
        color: 'white',
      },
    },
    y: {
      grid: {
        color: 'rgba(255, 255, 255, 0.3)',
      },
      ticks: {
        color: 'white',
      },
    },
  },
};

export function VerticalGraph({ data }) {
  return <Bar options={options} data={data} />;
}
