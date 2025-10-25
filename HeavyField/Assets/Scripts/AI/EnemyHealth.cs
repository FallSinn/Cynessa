using UnityEngine;

namespace HeavyField.EnemyAI
{
    public class EnemyHealth : MonoBehaviour
    {
        [SerializeField] private float maxHealth = 50f;
        [SerializeField] private GameObject deathEffect;

        public float CurrentHealth { get; private set; }

        private void Awake()
        {
            CurrentHealth = maxHealth;
        }

        public void TakeDamage(float amount)
        {
            CurrentHealth = Mathf.Max(0f, CurrentHealth - amount);
            if (CurrentHealth <= 0f)
            {
                Die();
            }
        }

        private void Die()
        {
            if (deathEffect)
            {
                Instantiate(deathEffect, transform.position, Quaternion.identity);
            }

            Destroy(gameObject);
        }
    }
}
