/**
 * Personal Finance Classifier - Frontend Interactions & Dynamic Charts
 */

document.addEventListener('DOMContentLoaded', () => {
  // Chart instances
  let spendingChartInstance = null;
  let budgetChartInstance = null;

  // Preset Configurations matching real online dataset
  const presets = {
    saver: {
      monthly_income: 48000,
      monthly_expenses: 28000,
      savings: 20000,
      loan_payments: 500,
      investment_amount: 1800,
      housing_utilities: 12000,
      food_dining: 8000,
      transportation: 3000,
      healthcare: 2000,
      entertainment: 1500,
      shopping_discretionary: 1500,
      credit_card_utilization: 15.0
    },
    balanced: {
      monthly_income: 42000,
      monthly_expenses: 31000,
      savings: 11000,
      loan_payments: 2500,
      investment_amount: 1500,
      housing_utilities: 14000,
      food_dining: 8500,
      transportation: 3500,
      healthcare: 2000,
      entertainment: 1500,
      shopping_discretionary: 1500,
      credit_card_utilization: 38.0
    },
    spender: {
      monthly_income: 40000,
      monthly_expenses: 36000,
      savings: 4000,
      loan_payments: 5000,
      investment_amount: 1200,
      housing_utilities: 16000,
      food_dining: 10000,
      transportation: 4000,
      healthcare: 2000,
      entertainment: 2000,
      shopping_discretionary: 2000,
      credit_card_utilization: 75.0
    }
  };

  // Slider update
  const ccSlider = document.getElementById('credit_card_utilization');
  const ccValue = document.getElementById('cc_value');
  if (ccSlider && ccValue) {
    ccSlider.addEventListener('input', (e) => {
      ccValue.textContent = `${e.target.value}%`;
    });
  }

  // Preset Button Listeners
  document.querySelectorAll('.btn-preset').forEach((btn) => {
    btn.addEventListener('click', () => {
      const type = btn.getAttribute('data-preset');
      const data = presets[type];
      if (!data) return;

      Object.keys(data).forEach((key) => {
        const input = document.getElementById(key);
        if (input) {
          input.value = data[key];
          if (key === 'credit_card_utilization' && ccValue) {
            ccValue.textContent = `${data[key]}%`;
          }
        }
      });

      // Automatically trigger classification
      const form = document.getElementById('financeForm');
      if (form) {
        form.dispatchEvent(new Event('submit'));
      }
    });
  });

  // Form Submission
  const form = document.getElementById('financeForm');
  if (form) {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();

      const submitBtn = document.getElementById('submitBtn');
      const originalText = submitBtn.innerHTML;
      submitBtn.innerHTML = '<span>⚡ Analyzing Behavior...</span>';
      submitBtn.disabled = true;

      const formData = {
        monthly_income: parseFloat(document.getElementById('monthly_income').value),
        monthly_expenses: parseFloat(document.getElementById('monthly_expenses').value),
        savings: parseFloat(document.getElementById('savings').value),
        loan_payments: parseFloat(document.getElementById('loan_payments').value),
        investment_amount: parseFloat(document.getElementById('investment_amount').value),
        housing_utilities: parseFloat(document.getElementById('housing_utilities').value),
        food_dining: parseFloat(document.getElementById('food_dining').value),
        transportation: parseFloat(document.getElementById('transportation').value),
        healthcare: parseFloat(document.getElementById('healthcare').value),
        entertainment: parseFloat(document.getElementById('entertainment').value),
        shopping_discretionary: parseFloat(document.getElementById('shopping_discretionary').value),
        credit_card_utilization: parseFloat(document.getElementById('credit_card_utilization').value),
        model_name: document.getElementById('model_name').value
      };

      try {
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(formData)
        });

        const result = await response.json();
        if (result.success) {
          renderResults(result.data, formData);
        } else {
          alert(`Analysis Error: ${result.error}`);
        }
      } catch (err) {
        console.error(err);
        alert('Network or server error while connecting to classifier.');
      } finally {
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
      }
    });
  }

  function renderResults(res, inputData) {
    const placeholder = document.getElementById('resultPlaceholder');
    const container = document.getElementById('resultContainer');
    if (placeholder) placeholder.style.display = 'none';
    if (container) container.style.display = 'block';

    // Archetype Banner
    const banner = document.getElementById('archetypeBanner');
    banner.className = `archetype-banner ${res.predicted_category}`;
    document.getElementById('archetypeTitle').textContent = res.predicted_category;
    document.getElementById('confidenceText').textContent = `${res.confidence}% Confidence (${res.model_used})`;

    // Health Score
    const gauge = document.getElementById('healthScoreGauge');
    const scoreVal = document.getElementById('healthScoreVal');
    scoreVal.textContent = res.health_score;

    let scoreColor = '#10b981'; // Green
    if (res.health_score < 45) {
      scoreColor = '#f43f5e'; // Red
    } else if (res.health_score < 70) {
      scoreColor = '#f59e0b'; // Amber
    } else if (res.health_score < 85) {
      scoreColor = '#3b82f6'; // Blue
    }
    gauge.style.borderColor = scoreColor;
    gauge.style.boxShadow = `0 0 20px ${scoreColor}55`;

    // Metrics boxes
    document.getElementById('metricSavingsRate').textContent = `${res.metrics.savings_rate}%`;
    document.getElementById('metricExpenseRatio').textContent = `${res.metrics.expense_to_income}%`;
    document.getElementById('metricDTI').textContent = `${res.metrics.debt_to_income}%`;
    document.getElementById('metricDiscretionary').textContent = `${res.metrics.discretionary_ratio}%`;

    // Render Alerts
    const alertsContainer = document.getElementById('alertsContainer');
    alertsContainer.innerHTML = '';
    if (res.alerts && res.alerts.length > 0) {
      res.alerts.forEach((alertMsg) => {
        const div = document.createElement('div');
        div.className = 'alert-item';
        div.innerHTML = `<span>⚠️</span><span>${alertMsg}</span>`;
        alertsContainer.appendChild(div);
      });
      alertsContainer.style.display = 'block';
    } else {
      alertsContainer.style.display = 'none';
    }

    // Render Recommendations
    const recsList = document.getElementById('recommendationsList');
    recsList.innerHTML = '';
    res.recommendations.forEach((rec) => {
      const card = document.createElement('div');
      card.className = 'rec-card';
      card.innerHTML = `
        <div class="rec-header">
          <span class="rec-title">${rec.title}</span>
          <span class="rec-tag">${rec.tag}</span>
        </div>
        <p class="rec-desc">${rec.description}</p>
      `;
      recsList.appendChild(card);
    });

    // Render Charts
    renderSpendingChart(inputData);
    renderBudgetComparisonChart(res.budget_comparison);

    // Scroll smoothly to result on smaller screens
    if (window.innerWidth < 960) {
      container.scrollIntoView({ behavior: 'smooth' });
    }
  }

  function renderSpendingChart(data) {
    const ctx = document.getElementById('spendingDoughnutChart');
    if (!ctx) return;

    if (spendingChartInstance) {
      spendingChartInstance.destroy();
    }

    spendingChartInstance = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Housing & Utils', 'Food & Dining', 'Transportation', 'Healthcare', 'Entertainment', 'Shopping'],
        datasets: [{
          data: [
            data.housing_utilities,
            data.food_dining,
            data.transportation,
            data.healthcare,
            data.entertainment,
            data.shopping_discretionary
          ],
          backgroundColor: [
            '#6366f1',
            '#3b82f6',
            '#06b6d4',
            '#10b981',
            '#f59e0b',
            '#f43f5e'
          ],
          borderWidth: 2,
          borderColor: '#111827'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'right',
            labels: {
              color: '#d1d5db',
              font: { size: 10 },
              boxWidth: 12
            }
          }
        },
        cutout: '68%'
      }
    });
  }

  function renderBudgetComparisonChart(budgetComp) {
    const ctx = document.getElementById('budgetBarChart');
    if (!ctx) return;

    if (budgetChartInstance) {
      budgetChartInstance.destroy();
    }

    const actual = budgetComp.actual;
    const benchmark = budgetComp.benchmark;

    budgetChartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['Needs (50%)', 'Wants (30%)', 'Savings (20%)'],
        datasets: [
          {
            label: 'Your Current Spending',
            data: [actual.needs, actual.wants, actual.savings_investments],
            backgroundColor: '#8b5cf6',
            borderRadius: 4
          },
          {
            label: 'Ideal 50/30/20 Target',
            data: [benchmark.needs, benchmark.wants, benchmark.savings_investments],
            backgroundColor: 'rgba(255, 255, 255, 0.15)',
            borderColor: 'rgba(255, 255, 255, 0.4)',
            borderWidth: 1,
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            ticks: { color: '#9ca3af', font: { size: 10 } },
            grid: { display: false }
          },
          y: {
            ticks: { color: '#9ca3af', font: { size: 10 } },
            grid: { color: 'rgba(255, 255, 255, 0.05)' }
          }
        },
        plugins: {
          legend: {
            position: 'top',
            labels: { color: '#d1d5db', font: { size: 10 }, boxWidth: 12 }
          }
        }
      }
    });
  }
});
