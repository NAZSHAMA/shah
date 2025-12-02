document.addEventListener('DOMContentLoaded', () => {
    const teacherGrid = document.getElementById('teacher-grid');

    // Fetch teacher data from the mock API (teachers.json)
    fetch('teachers.json')
        .then(response => response.json())
        .then(data => {
            const teachers = data.teachers;
            teachers.forEach(teacher => {
                const card = document.createElement('div');
                card.className = 'teacher-card';

                card.innerHTML = `
                    <img src="${teacher.image}" alt="Photo of ${teacher.name}">
                    <div class="teacher-card-content">
                        <h3>${teacher.name}</h3>
                        <p class="specialization">${teacher.specialization}</p>
                        <p class="experience"><strong>Experience:</strong> ${teacher.experience}</p>
                        <p>${teacher.bio}</p>
                        <a href="mailto:contact@specialedconnect.com?subject=Inquiry about ${teacher.name}" class="contact-button">Contact ${teacher.name}</a>
                    </div>
                `;
                teacherGrid.appendChild(card);
            });
        })
        .catch(error => {
            console.error('Error fetching teacher data:', error);
            teacherGrid.innerHTML = '<p>Could not load teacher information. Please try again later.</p>';
        });
});
