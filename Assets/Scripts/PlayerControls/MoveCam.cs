using UnityEngine;

public class MoveCam : MonoBehaviour
{
    public Transform camPos;

    [Header("Debug")]
    public bool debugMode = false;

    // Move the camera after all updates to reduce jitter
    void LateUpdate()
    {
        if (camPos == null)
        {
            if (debugMode) Debug.LogWarning("[MoveCam] camPos is null.");
            return;
        }

        // Compute distance before moving (for a meaningful debug check)
        float dist = Vector3.Distance(transform.position, camPos.position);
        if (debugMode && dist > 1f)
            Debug.LogWarning($"[MoveCam] large camera teleport distance: {dist:F2}");

        transform.position = camPos.position;
    }
}
