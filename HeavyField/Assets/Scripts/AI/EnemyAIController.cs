using UnityEngine;
using UnityEngine.AI;
using HeavyField.Player;

namespace HeavyField.EnemyAI
{
    [RequireComponent(typeof(NavMeshAgent))]
    public class EnemyAIController : MonoBehaviour
    {
        private enum State
        {
            Idle,
            Patrol,
            Chase,
            Attack
        }

        [SerializeField] private Transform[] patrolPoints;
        [SerializeField] private float detectionRadius = 20f;
        [SerializeField] private float attackRadius = 10f;
        [SerializeField] private float attackCooldown = 1.5f;
        [SerializeField] private float attackDamage = 10f;
        [SerializeField] private float lookHeightOffset = 1.5f;

        private State currentState;
        private int currentPatrolIndex;
        private NavMeshAgent agent;
        private PlayerHealth player;
        private float lastAttackTime;

        private void Awake()
        {
            agent = GetComponent<NavMeshAgent>();
            player = FindObjectOfType<PlayerHealth>();
            currentState = patrolPoints != null && patrolPoints.Length > 0 ? State.Patrol : State.Idle;
        }

        private void Update()
        {
            if (!player)
            {
                currentState = State.Idle;
                return;
            }

            float distance = Vector3.Distance(transform.position, player.transform.position);
            bool canSeePlayer = distance <= detectionRadius && HasLineOfSight();

            switch (currentState)
            {
                case State.Idle:
                    if (canSeePlayer)
                    {
                        currentState = State.Chase;
                    }
                    break;
                case State.Patrol:
                    Patrol();
                    if (canSeePlayer)
                    {
                        currentState = State.Chase;
                    }
                    break;
                case State.Chase:
                    Chase();
                    if (!canSeePlayer)
                    {
                        currentState = patrolPoints.Length > 0 ? State.Patrol : State.Idle;
                    }
                    else if (distance <= attackRadius)
                    {
                        currentState = State.Attack;
                    }
                    break;
                case State.Attack:
                    Attack();
                    if (!canSeePlayer)
                    {
                        currentState = patrolPoints.Length > 0 ? State.Patrol : State.Idle;
                    }
                    else if (distance > attackRadius * 1.25f)
                    {
                        currentState = State.Chase;
                    }
                    break;
            }
        }

        private void Patrol()
        {
            if (patrolPoints == null || patrolPoints.Length == 0)
            {
                return;
            }

            if (!agent.pathPending && agent.remainingDistance < 0.5f)
            {
                currentPatrolIndex = (currentPatrolIndex + 1) % patrolPoints.Length;
                agent.SetDestination(patrolPoints[currentPatrolIndex].position);
            }
        }

        private void Chase()
        {
            if (!player)
            {
                return;
            }

            agent.SetDestination(player.transform.position);
        }

        private void Attack()
        {
            if (Time.time < lastAttackTime + attackCooldown)
            {
                return;
            }

            if (!player)
            {
                return;
            }

            agent.SetDestination(transform.position);
            transform.LookAt(player.transform.position + Vector3.up * lookHeightOffset);
            player.TakeDamage(attackDamage);
            lastAttackTime = Time.time;
        }

        private bool HasLineOfSight()
        {
            Vector3 origin = transform.position + Vector3.up * lookHeightOffset;
            Vector3 target = player.transform.position + Vector3.up * 1.5f;
            if (Physics.Raycast(origin, (target - origin).normalized, out RaycastHit hit, detectionRadius))
            {
                return hit.collider.GetComponent<PlayerHealth>() != null;
            }

            return false;
        }
    }
}
