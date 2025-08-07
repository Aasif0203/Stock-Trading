import React from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { Doughnut } from 'react-chartjs-2';

ChartJS.register(ArcElement, Tooltip, Legend);

export function DoughnutChart({data, width = 300, height = 300}) {
  // const options = {
  //   responsive: true,
  //   maintainAspectRatio: false,
  //   plugins: {
  //     legend: {
  //       position: 'bottom',
  //       labels: {
  //         font: {
  //           size: 12
  //         }
  //       }
  //     }
  //   }
  // };

  return (
    <div style={{ width:'95%', height:'600px' }}>
      <Doughnut data={data} />
    </div>
  );
}