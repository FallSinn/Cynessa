using System;
using UnityEngine;
using HeavyField.Player;

namespace HeavyField.Weapons
{
    public class Bullet : MonoBehaviour
    {
        [SerializeField] private float maxLifetime = 5f;
        [SerializeField] private float skinWidth = 0.1f;

        private float damage;
        private float speed;
        private float lifetime;
        private int shooterLayer;

        public Action OnReturnedToPool;

        public void Configure(float damageValue, float speedValue, int layer)
        {
            damage = damageValue;
            speed = speedValue;
            shooterLayer = layer;
            lifetime = 0f;
        }

        public void OnSpawned()
        {
            lifetime = 0f;
        }

        private void Update()
        {
            float distance = speed * Time.deltaTime;
            Vector3 direction = transform.forward;
            if (Physics.Raycast(transform.position, direction, out RaycastHit hit, distance + skinWidth, Physics.DefaultRaycastLayers, QueryTriggerInteraction.Ignore))
            {
                HandleHit(hit.collider, hit.point);
                return;
            }

            transform.position += direction * distance;
            lifetime += Time.deltaTime;
            if (lifetime >= maxLifetime)
            {
                OnReturnedToPool?.Invoke();
            }
        }

        private void HandleHit(Collider other, Vector3 point)
        {
            if (other.gameObject.layer == shooterLayer)
            {
                return;
            }

            if (other.TryGetComponent(out PlayerHealth playerHealth))
            {
                playerHealth.TakeDamage(damage);
            }

            EnemyAI.EnemyHealth enemyHealth = other.GetComponentInParent<EnemyAI.EnemyHealth>();
            if (enemyHealth)
            {
                enemyHealth.TakeDamage(damage);
            }

            // TODO: add impact effects.
            OnReturnedToPool?.Invoke();
        }
    }
}
