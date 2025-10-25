using UnityEngine;
using UnityEngine.Events;

namespace HeavyField.Player
{
    public class PlayerHealth : MonoBehaviour
    {
        [SerializeField] private float maxHealth = 100f;

        public UnityEvent onDeath;
        public UnityEvent<float> onHealthChanged;

        public float CurrentHealth { get; private set; }

        private void Awake()
        {
            CurrentHealth = maxHealth;
        }

        public void TakeDamage(float amount)
        {
            if (CurrentHealth <= 0f)
            {
                return;
            }

            CurrentHealth = Mathf.Max(0f, CurrentHealth - amount);
            onHealthChanged?.Invoke(CurrentHealth);

            if (CurrentHealth <= 0f)
            {
                onDeath?.Invoke();
            }
        }

        public void Heal(float amount)
        {
            if (CurrentHealth <= 0f)
            {
                return;
            }

            CurrentHealth = Mathf.Clamp(CurrentHealth + amount, 0f, maxHealth);
            onHealthChanged?.Invoke(CurrentHealth);
        }
    }
}
